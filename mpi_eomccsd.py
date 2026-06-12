from functools import reduce
import copy
import heapq
import os
import resource
import time
import warnings

import numpy as np
import psutil
import scipy.linalg
from mpi4py import MPI
from numpy.lib.mixins import NDArrayOperatorsMixin
from pyscf import lib
from pyscf.lib import logger, misc
import pyscf.pbc.cc.kccsd_rhf
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.cc.kccsd_rhf import (
    _get_epq,
    nested_to_vector,
    vector_to_nested,
)
from pyscf.pbc.df import df
from pyscf.pbc.lib.kpts import KQuartets, MORotationMatrix
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.mpitools.mpi_blksize import get_max_blocksize_from_mem
from pyscf.pbc.mpitools.mpi_helper import generate_task_list, safeAllreduceInPlace
from pyscf.pbc.mp.kmp2 import (
    get_nocc,
    padded_mo_coeff,
    padding_k_idx,
)

np.set_printoptions(precision=16, suppress=True, floatmode="fixed", threshold=np.inf)

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD
nphase = size

_PROCESS = psutil.Process(os.getpid())
def report_mem(msg=""):
    """Print current and peak resident memory for this MPI rank."""
    rss = _PROCESS.memory_info().rss / 1024**2
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{msg:<24s} | Current RSS: {rss:10.2f} MB | Peak: {peak:10.2f} MB | Rank: {rank}")
def get_mem():
    """Return current and peak resident memory for this MPI rank."""
    rss = _PROCESS.memory_info().rss / 1024**2
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return f"Current RSS: {rss:10.2f} MB | Peak: {peak:10.2f} MB"
def einsum(*args):
    return np.einsum(*args, optimize="optimal")

_MPI_BLOCK_DTYPE_CACHE = {}

def _mpi_scalar_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        if dtype.itemsize != np.dtype(np.complex128).itemsize:
            raise TypeError(f"Unsupported complex MPI dtype: {dtype}")
        return MPI.C_DOUBLE_COMPLEX
    if np.issubdtype(dtype, np.floating):
        if dtype.itemsize == np.dtype(np.float64).itemsize:
            return MPI.DOUBLE
        if dtype.itemsize == np.dtype(np.float32).itemsize:
            return MPI.FLOAT
    raise TypeError(f"Unsupported MPI dtype: {dtype}")

def _mpi_block_dtype(dtype, block_size):
    key = (np.dtype(dtype).str, int(block_size))
    block_dt = _MPI_BLOCK_DTYPE_CACHE.get(key)
    if block_dt is None:
        block_dt = _mpi_scalar_dtype(dtype).Create_contiguous(int(block_size))
        block_dt.Commit()
        _MPI_BLOCK_DTYPE_CACHE[key] = block_dt
    return block_dt

def _get_work_buffer(indices_map, key, shape, dtype):
    dtype = np.dtype(dtype)
    shape = tuple(int(x) for x in shape)
    size_needed = int(np.prod(shape, dtype=np.int64))
    cache = indices_map.setdefault("_mpi_work_buffers", {})
    buf = cache.get(key)
    if buf is None or buf.dtype != dtype or buf.size < size_needed:
        buf = np.empty(max(size_needed, 1), dtype=dtype)
        cache[key] = buf
    return buf[:size_needed].reshape(shape)

def _direct_ibz_block(arr, kqrts, m):
    m = int(m)
    if isinstance(arr, KsymmArray):
        if arr._m2loc is not None:
            loc = arr._m2loc.get(m)
            if loc is not None:
                return arr.data[loc]
        else:
            return arr.data[m]
    ki, kj, ka, _ = kqrts.kqrts_ibz[m]
    return arr[ki, kj, ka]

def _add_direct_ibz_block(arr, kqrts, m, value):
    m = int(m)
    if isinstance(arr, KsymmArray):
        if arr._m2loc is not None:
            loc = arr._m2loc.get(m)
            if loc is not None:
                arr.data[loc] += value
                return
        else:
            arr.data[m] += value
            return
    ki, kj, ka, _ = kqrts.kqrts_ibz[m]
    arr[ki, kj, ka] += value

##############################################################
##############################################################
# DIIS
class MPIDIIS(lib.diis.DIIS):
   
    def _store(self, key, value):
        if self._diisfile is None:
            if isinstance(self.filename, str):
                filename = self.filename + '__rank' + str(rank)
                self._diisfile = lib.H5TmpFile(filename, 'w')

            elif not (self.incore or value.size < lib.diis.INCORE_SIZE):
                self._diisfile = lib.H5TmpFile(self.filename, 'w')

        return lib.diis.DIIS._store(self, key, value)
    
    def update(self, x, xerr=None):
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        dt = np.asarray(self.get_err_vec(self._head-1))
        if self._H is None:
            self._H = np.zeros((self.space+1,self.space+1), dt.dtype)
            self._H[0,1:] = self._H[1:,0] = 1
        for i in range(nd):
            tmp_local = 0
            dti = self.get_err_vec(i)
            if rank == 0:
                for p0, p1 in misc.prange(0, self.t1_size, lib.diis.BLOCK_SIZE):
                    tmp_local += np.dot(dt[p0:p1].conj(), dti[p0:p1])
            for p0, p1 in misc.prange(self.t1_size, dt.size, lib.diis.BLOCK_SIZE):
                tmp_local += np.dot(dt[p0:p1].conj(), dti[p0:p1])
            tmp = comm.allreduce(tmp_local, op=MPI.SUM)
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conjugate()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = self.extrapolate(nd)

            self._store('xprev', xnew)
            if 'xprev' not in self._buffer:  # not incore
                self._xprev = self._diisfile['xprev']
        return xnew.reshape(x.shape)
    
    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError('No vector found in DIIS object.')

        h = self._H[:nd+1,:nd+1]
        g = np.zeros(nd+1, h.dtype)
        g[0] = 1

        if rank == 0:
            w, v = scipy.linalg.eigh(h)
            if np.any(abs(w)<1e-14):
                logger.debug(self, 'Linear dependence found in DIIS error vectors.')
                idx = abs(w)>1e-14
                c = np.dot(v[:,idx]*(1./w[idx]), np.dot(v[:,idx].T.conj(), g))
            else:
                try:
                    c = np.linalg.solve(h, g)
                except np.linalg.linalg.LinAlgError as e:
                    logger.warn(self, ' diis singular, eigh(h) %s', w)
                    raise e
        else:
            c = None
        c = comm.bcast(c, root=0)
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = np.zeros(xi.size, c.dtype)
            for p0, p1 in misc.prange(0, xi.size, lib.diis.BLOCK_SIZE):
                xnew[p0:p1] += xi[p0:p1] * ci
        return xnew

    def restore(self, filename, inplace=True):
        filename_base = filename.split('__rank')[0]
        filename = filename_base + '__rank' + str(rank)
        val = lib.diis.DIIS.restore(self, filename, inplace)
        if inplace:
            self.filename = filename_base
        return val

def restore(filename):
    return MPIDIIS().restore(filename)

##############################################################
##############################################################
# KsymmArray
def s2_index(A):
    A = np.asarray(A)
    mp = {tuple(row): i for i, row in enumerate(A)}
    partner = np.fromiter((mp.get(tuple(row[[1,0,3,2]]), -1) for row in A), dtype=np.int64, count=A.shape[0])
    i = np.arange(A.shape[0])
    mask = (partner >= 0) & (i < partner)
    return partner[mask]

def empty(shape, dtype=float, order='C', metadata=None):
    if metadata is None:
        return np.empty(shape, dtype, order)
    else:
        return KsymmArray(shape, dtype, order, metadata)

def empty_like(a, *args, **kwargs):
    if isinstance(a, KsymmArray):
        return KsymmArray(a.subarray_shape,
                          a.dtype,
                          a.subarray_order,
                          a.metadata)
    else:
        return np.empty_like(a)
    
def zeros_like(a, *args, **kwargs):
    if isinstance(a, KsymmArray):
        return KsymmArray(a.subarray_shape,
                          a.dtype,
                          a.subarray_order,
                          a.metadata,
                          init_with_zeros=True)
    else:
        return np.zeros_like(a)


class KsymmArray(NDArrayOperatorsMixin):
    """Symmetry-aware storage for k-point tensors.

    The backing storage keeps only irreducible blocks and reconstructs symmetry
    partners on access with the supplied rotation matrices.  Four-index tensors
    can also be restricted to locally owned irreducible rows for MPI work
    partitioning.
    """

    def __init__(self, subarray_shape, dtype=float, subarray_order='C', metadata=None,
                 init_with_zeros=False):
        self.metadata = {} if metadata is None else metadata
        self._subarray_shape = list(subarray_shape)
        self._subarray_ndim = len(subarray_shape)
        self._subarray_order = subarray_order
        self._dtype = np.dtype(dtype)
        
        self._owned_m = None
        self._m2loc = None
        if self._subarray_ndim == 4:
            owned_m = metadata.get('owned_m', None)
            if owned_m is not None:
                self._owned_m = np.asarray(owned_m, dtype=int)
                self._m2loc = {m: i for i, m in enumerate(self._owned_m.tolist())}
        
        incore = metadata.get('incore', True)
        self._datafile = None
        self.data = self._init(subarray_order, incore, init_with_zeros)

    def _init(self, order, incore=True, init_with_zeros=False):
        if self.subarray_ndim == 2:
            kpts = self.metadata['kpts']
            n_subarray = kpts.nkpts_ibz
        elif self.subarray_ndim == 4:
            kqrts = self.metadata['kqrts']
            if self._owned_m is not None:
                n_subarray = len(self._owned_m)
            else:
                n_subarray = len(kqrts.kqrts_ibz)
        else:
            raise NotImplementedError

        data = None
        shape = [n_subarray,] + self.subarray_shape
        if incore:
            if init_with_zeros:
                fn_init = np.zeros
            else:
                fn_init = np.empty
            if order == 'C':
                data = fn_init(shape, self.dtype, order)
            else:
                data = []
                for i in range(n_subarray):
                    data.append(fn_init(self.subarray_shape, self.dtype, order))
                data = np.asarray(data, order='K')
        else:
            prefix = self.metadata.get('prefix', f'ksymm_{self.metadata.get("label","arr")}')
            shape = [n_subarray,] + self.subarray_shape
            if self.subarray_ndim == 4:
                n1,n2,n3,n4 = self.subarray_shape  # = nocc,nocc,nvir,nvir
                tile_v = min(64, n3, n4)           # ~32–128 is fine
                chunks = (1, n1, n2, tile_v, tile_v)
            elif self.subarray_ndim == 2:
                n1,n2 = self.subarray_shape        # e.g. (nocc,nvir)
                chunks = (1, n1, min(256, n2))
            else:
                chunks = None
            
            self._datafile = lib.H5TmpFile(prefix=prefix)
            data = self._datafile.create_dataset('data', shape, self.dtype.char, chunks=chunks)
        return data

    @property
    def shape(self):
        nkpts = self.metadata['kpts'].nkpts
        nk = [nkpts,] * (self.subarray_ndim-1)
        return tuple(nk + list(self.subarray_shape))

    @property
    def ndim(self):
        return self.subarray_ndim-1 + self.subarray_ndim

    @property
    def subarray_ndim(self):
        return self._subarray_ndim

    @property
    def subarray_shape(self):
        return self._subarray_shape

    @property
    def subarray_order(self):
        return self._subarray_order

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        if self.subarray_ndim == 2:
            return self._getitem_2d(key)
        elif self.subarray_ndim == 4:
            return self._getitem_4d(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        if self.subarray_ndim == 2:
            return self._setitem_2d(key, value)
        elif self.subarray_ndim == 4:
            return self._setitem_4d(key, value)
        else:
            raise NotImplementedError

    def _getitem_2d(self, key):
        kpts = self.metadata['kpts']
        rmat = self.metadata['rmat']
        label = self.metadata['label']
        trans = self.metadata['trans']
        if isinstance(key, (int, np.integer)):
            return transform_2d(self.data, kpts, key, rmat, label, trans)
        elif isinstance(key, (slice, np.ndarray)):
            data = []
            for ki in np.arange(kpts.nkpts)[key]:
                data.append(transform_2d(self.data, kpts, ki, rmat, label, trans))
            return np.asarray(data)
        else:
            raise NotImplementedError

    def _getitem_4d(self, key):
        kpts = self.metadata['kpts']
        kqrts = self.metadata['kqrts']
        rmat = self.metadata['rmat']
        label = self.metadata['label']
        trans = self.metadata['trans']

        shape = [kpts.nkpts,] * 3
        coords = index_to_coords(key, shape)
        
        if self._owned_m is not None:
            def get_ibz_block(m: int):
                if m in self._m2loc:
                    return self.data[self._m2loc[m]]
                else:
                    raise KeyError(f'IBZ row {m} not local!')
            def local_transform_from_ibz(klc):
                kk_bz  = kpts.ktuple_to_index(klc)
                kk_ibz = kqrts.bz2ibz[kk_bz]
                i,j,a,b = kqrts.kqrts_ibz[kk_ibz]
                if (i,j,a) == tuple(klc):
                    return get_ibz_block(kk_ibz)
                ibz_block = get_ibz_block(kk_ibz)
                return _rotate_ibz_block_to_bz(ibz_block, kpts, kqrts, kk_bz, rmat, label, trans, i, j, a, b)
            if coords.ndim == 1:
                return local_transform_from_ibz(coords)
            else:
                return np.asarray([local_transform_from_ibz(klc) for klc in coords])
            
            
        if coords.ndim == 1:
            klc = coords
            return transform_4d(self.data, kpts, kqrts, klc, rmat, label, trans)
        else:
            data = []
            for klc in coords:
                data.append(transform_4d(self.data, kpts, kqrts, klc, rmat, label, trans))
            return np.asarray(data)

    def _setitem_2d(self, key, value):
        kpts = self.metadata['kpts']
        #TODO allow broadcasting
        value = value.reshape(-1, *self.subarray_shape)
        if isinstance(key, (int, np.integer)):
            set_2d(self.data, value, kpts, key)
        elif isinstance(key, (slice, np.ndarray)):
            ki = np.arange(kpts.nkpts)[key]
            set_2d(self.data, value, kpts, ki)
        else:
            raise NotImplementedError

    def _setitem_4d(self, key, value):
        kpts = self.metadata['kpts']
        kqrts = self.metadata['kqrts']
        #TODO allow broadcasting
        value = value.reshape(-1, *self.subarray_shape)

        shape = [kpts.nkpts,] * 3
        coords = index_to_coords(key, shape)
        set_4d(self.data, value, kpts, kqrts, coords, self._m2loc)

    def todense(self):
        #TODO allow to return a hdf5 dataset
        return self[:].reshape(self.shape)

    @staticmethod
    def fromdense(arr, shape, dtype=None, order=None, metadata=None):
        if dtype is None:
            dtype = arr.dtype
        order = _guess_input_order(arr, order)
        if metadata is None:
            raise RuntimeError('metadata not initialized')

        out = KsymmArray(shape, dtype, order, metadata)
        arr = arr.reshape(out.shape)
        if out.subarray_ndim == 2:
            kpts = out.metadata['kpts']
            for ki in kpts.ibz2bz:
                ki_ibz = kpts.bz2ibz[ki]
                out[ki_ibz] = arr[ki]
        elif out.subarray_ndim == 4:
            kqrts = out.metadata['kqrts']
            for m, kq in enumerate(kqrts.kqrts_ibz):
                ki, kj, ka, kb = kq
                out[m] = arr[ki, kj, ka]
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def fromraw(arr, shape, dtype=None, order=None, metadata=None):
        if dtype is None:
            dtype = arr.dtype
        order = _guess_input_order(arr, order)
        if metadata is None:
            raise RuntimeError('metadata not initialized')

        out = KsymmArray(shape, dtype, order, metadata)
        arr = arr.reshape(-1, *out.subarray_shape)
        for i, a in enumerate(arr):
            out.data[i] = np.asarray(a, dtype=dtype, order=order)
        return out

    @staticmethod
    def zeros(shape, dtype=float, order='C', metadata=None):
        out = KsymmArray(shape, dtype, order, metadata, init_with_zeros=True)
        return out

def _rotate_ibz_block_to_bz(ibz_block, kpts, kqrts, kk_bz, rmat, label, trans, i, j, a, b):
    pi, pj, pa, pb = label
    rmat_i = getattr(rmat, pi*2)
    rmat_j = getattr(rmat, pj*2)
    rmat_a = getattr(rmat, pa*2)
    rmat_b = getattr(rmat, pb*2)

    iop = kqrts.stars_ops_bz[kk_bz]
    rot_i = rmat_i[i][iop]
    rot_j = rmat_j[j][iop]
    rot_a = rmat_a[a][iop]
    rot_b = rmat_b[b][iop]

    ti, tj, ta, tb = trans
    if ti == 'c': rot_i = rot_i.conj()
    if tj == 'c': rot_j = rot_j.conj()
    if ta == 'c': rot_a = rot_a.conj()
    if tb == 'c': rot_b = rot_b.conj()

    di, dj, da, db = ibz_block.shape
    tmp = np.dot(rot_i.T, ibz_block.reshape(di,-1))
    tmp = tmp.reshape(di,dj,-1).transpose(1,0,2)
    tmp = np.dot(rot_j.T, tmp.reshape(dj,-1))
    tmp = tmp.reshape(dj,di,-1).transpose(1,0,2)
    tmp = tmp.reshape(-1,da,db).transpose(0,2,1).reshape(-1,da)
    tmp = np.dot(tmp, rot_a)
    tmp = tmp.reshape(-1,db,da).transpose(0,2,1).reshape(-1,db)
    out = np.dot(tmp, rot_b).reshape(di,dj,da,db)
    return out

def _guess_input_order(arr, order=None):
    if order is None:
        order = 'C'
        if isinstance(arr, np.ndarray):
            if arr.flags.c_contiguous:
                order = 'C'
            elif arr.flags.f_contiguous:
                order = 'F'
    elif order not in ('C', 'F'):
        order = 'C'
    return order

def set_2d(arr, value, kpts, ki):
    if isinstance(ki, (int, np.integer)):
        ki = np.asarray([ki,])

    mask = np.isin(ki, kpts.ibz2bz)
    if not mask.all():
        warnings.warn(f'Indices {ki[~mask]} are not in the irreducible wedge. '
                       'The corresponding data will be discarded.')

    ki_ibz = kpts.bz2ibz[ki[mask]]
    arr[ki_ibz] = value[mask]

def set_4d(arr, value, kpts, kqrts, klc, m2loc=None):
    klc = klc.reshape(-1, 3)
    kk_bz = [kpts.ktuple_to_index(s) for s in klc]
    kk_bz = np.asarray(kk_bz)

    mask = np.isin(kk_bz, kqrts.ibz2bz)
    if not mask.all():
        kk_tmp = [kpts.index_to_ktuple(k, 3) for k in kk_bz[~mask]]
        warnings.warn(f'Indices {kk_tmp} are not in the irreducible wedge. '
                       'The corresponding data will be discarded.')

    kk_ibz = kqrts.bz2ibz[kk_bz[mask]]
    if m2loc is None:
        arr[kk_ibz] = value[mask]
    else:
        arr[np.fromiter((m2loc[m] for m in kk_ibz), dtype=int)] = value[mask]

def transform_2d(arr, kpts, ki, rmat, label, trans):
    ki_ibz = kpts.bz2ibz[ki]
    ki_ibz_bz = kpts.ibz2bz[ki_ibz]
    if ki == ki_ibz_bz:
        return arr[ki_ibz]

    pi, pj = label
    rmat_i = getattr(rmat, pi*2)
    rmat_j = getattr(rmat, pj*2)

    iop = kpts.stars_ops_bz[ki]
    rot_i = rmat_i[ki_ibz_bz][iop]
    rot_j = rmat_j[ki_ibz_bz][iop]
    ti, tj = trans
    if ti == 'c':
        rot_i = rot_i.conj()
    if tj == 'c':
        rot_j = rot_j.conj()

    #:einsum('ia,ij,ab->jb', arr[ki_ibz], rot_i, rot_j)
    out = reduce(np.dot, (rot_i.T, arr[ki_ibz], rot_j))
    return out

def transform_4d(arr, kpts, kqrts, klc, rmat, label, trans):
    kk_bz = kpts.ktuple_to_index(klc)
    kk_ibz = kqrts.bz2ibz[kk_bz]
    i,j,a,b = kqrts.kqrts_ibz[kk_ibz]
    if (i,j,a) == tuple(klc):
        return arr[kk_ibz]

    pi, pj, pa, pb = label
    rmat_i = getattr(rmat, pi*2)
    rmat_j = getattr(rmat, pj*2)
    rmat_a = getattr(rmat, pa*2)
    rmat_b = getattr(rmat, pb*2)

    iop = kqrts.stars_ops_bz[kk_bz]
    rot_i = rmat_i[i][iop]
    rot_j = rmat_j[j][iop]
    rot_a = rmat_a[a][iop]
    rot_b = rmat_b[b][iop]

    ti, tj, ta, tb = trans
    if ti == 'c':
        rot_i = rot_i.conj()
    if tj == 'c':
        rot_j = rot_j.conj()
    if ta == 'c':
        rot_a = rot_a.conj()
    if tb == 'c':
        rot_b = rot_b.conj()

    di, dj, da, db = arr[kk_ibz].shape
    #:einsum('ijab,ik,jl,ac,bd->klcd', arr[kk_ibz],
    #        rot_i, rot_j, rot_a, rot_b)

    #tmp = np.einsum('ik,ijab->jkab', rot_i, arr[kk_ibz])
    tmp = np.dot(rot_i.T, arr[kk_ibz].reshape(di,-1)) #k,jab
    tmp = tmp.reshape(di,dj,-1).transpose(1,0,2) #j,k,ab

    #tmp = np.einsum('jkab,jl->klab', tmp, rot_j)
    tmp = np.dot(rot_j.T, tmp.reshape(dj,-1)) #l,kab
    tmp = tmp.reshape(dj,di,-1).transpose(1,0,2) #k,l,ab

    #tmp = np.einsum('klab,ac->klbc', tmp, rot_a)
    tmp = tmp.reshape(-1,da,db).transpose(0,2,1).reshape(-1,da) #klb,a
    tmp = np.dot(tmp, rot_a) #klb,c

    #out = np.einsum('klbc,bd->klcd', tmp, rot_b)
    tmp = tmp.reshape(-1,db,da).transpose(0,2,1).reshape(-1,db) #klc,b
    out = np.dot(tmp, rot_b).reshape(di,dj,da,db) #k,l,c,d
    return out

def index_to_coords(key, shape):
    if not isinstance(key, tuple):
        key = (key,)

    idxs = []
    for i, k in enumerate(key):
        n = shape[i]
        if isinstance(k, slice):
            idx = slice_to_coords(k, n)
        elif isinstance(k, (int, np.integer)):
            idx = [k,]
        elif isinstance(k, np.ndarray) and k.ndim == 1:
            idx = k
        else:
            raise NotImplementedError
        idxs.append(idx)

    ndim = len(shape)
    if len(idxs) > ndim:
        raise RuntimeError
    elif len(idxs) < ndim:
        for i in range(len(idxs),ndim):
            n = shape[i]
            idxs.append(np.arange(n))

    coords = lib.cartesian_prod(idxs)
    if all(isinstance(k, (int, np.integer)) for k in key) and len(key) == ndim:
        coords = coords[0]
    return coords

def slice_to_coords(k, n):
    start, stop, step = k.start, k.stop, k.step
    if start is None:
        start = 0
    elif start < 0:
        start += n
    if stop is None:
        stop = n
    elif stop < 0:
        stop += n
    if step is None:
        step = 1
    return np.arange(start, stop, step)


zeros = KsymmArray.zeros
fromraw = KsymmArray.fromraw
fromdense = KsymmArray.fromdense

##############################################################
##############################################################
# MPI utils
def safeAllreduceInPlace_ksymm(comm, ksymm):
    MEM_SIZE = 0.5e9
    data = ksymm.data  
    kqrts = ksymm.metadata['kqrts']
    n_subarray = len(kqrts.kqrts_ibz)
    shape = [n_subarray,] + ksymm.subarray_shape
    length = len(shape)
    chunk_size = get_max_blocksize_from_mem(list(shape),16.,MEM_SIZE,priority_list=np.arange(length)[::-1])
    task_list = generate_task_list(chunk_size, shape)
    
    for block in task_list:
        which_slice = [slice(*x) for x in block]
        tmp = data[tuple(which_slice)]
        comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
        
def safeBcast(comm, ksymm, root=0):
    buf = np.ascontiguousarray(ksymm.data)
    flat = buf.reshape(-1)
    chunk_elems = (1 << 31) // 2 
    for i in range(0, flat.size, chunk_elems):
        comm.Bcast(flat[i:i + chunk_elems], root=root)
    if buf is not ksymm.data:
        ksymm.data[...] = buf
        
def build_indirect_map(kpts, kqrts):
    nk = kpts.nkpts
    indirect = np.empty((nk, nk, nk), dtype=bool)
    for kk in range(nk):
        for kl in range(nk):
            for kc in range(nk):
                klc = (kk, kl, kc)
                kk_bz  = kpts.ktuple_to_index(klc)
                kk_ibz = kqrts.bz2ibz[kk_bz]
                i, j, a, _ = kqrts.kqrts_ibz[kk_ibz]
                indirect[kk,kl,kc] = (i, j, a) != klc
    return indirect
        
def plan_tasks(kqrts_ibz, kpts, kqrts, nranks, cnt_funct, indirect_map, nphase=1, return_owner=False):
    weights = np.empty(len(kqrts_ibz), dtype=int)
    owner = np.empty(len(kqrts_ibz), dtype=int)
    for i, kq in enumerate(kqrts_ibz):
        w = cnt_funct(kq, kpts, kqrts, indirect_map)
        weights[i] = w
    base = len(kqrts_ibz) // nranks
    extra = len(kqrts_ibz) % nranks
    quotas = np.array([base + (1 if r < extra else 0) for r in range(nranks)], dtype=int)
    order = np.argsort(-weights)
    heap = [(0, r) for r in range(nranks) if quotas[r] > 0]
    heapq.heapify(heap)
    assignments = [[] for _ in range(nranks)]
    load = np.zeros(nranks, dtype=int)
    for idx in order:
        cur_load, r = heapq.heappop(heap)
        assignments[r].append(idx)
        owner[idx] = r // nphase
        new_load = cur_load + weights[idx]
        load[r] = new_load
        quotas[r] -= 1
        if quotas[r] > 0:
            heapq.heappush(heap, (new_load, r))
    if return_owner:
        return assignments, load, weights, owner
    return assignments, load, weights

def exchange_data(request_map, name, out_map, nphase_tmp=1):
    out_map[f"{name}_send_idx_buf"] = []
    out_map[f"{name}_send_cnt"] = []
    out_map[f"{name}_send_offset"] = []
    out_map[f"{name}_send_cnt32"] = []
    out_map[f"{name}_send_offset32"] = []
    out_map[f"{name}_recv_idx_buf"] = []
    out_map[f"{name}_recv_cnt"] = []
    out_map[f"{name}_recv_offset"] = []
    out_map[f"{name}_recv_cnt32"] = []
    out_map[f"{name}_recv_offset32"] = []
    for i in range(nphase_tmp):
        recv_idx = [np.fromiter(sorted(request_map[i][r]), dtype=int) for r in range(size)]
        recv_cnt = np.array([len(x) for x in recv_idx], dtype=int)
        send_cnt = np.empty(size, dtype=int)
        comm.Alltoall(recv_cnt, send_cnt)
        recv_idx_buf = np.concatenate(recv_idx, axis=0)
        send_idx_size = int(np.sum(send_cnt))
        send_idx_buf = np.empty(send_idx_size, dtype=int)
        recv_offset = np.zeros_like(recv_cnt)
        np.cumsum(recv_cnt[:-1], axis=0, dtype=int, out=recv_offset[1:])
        send_offset = np.zeros_like(send_cnt)
        np.cumsum(send_cnt[:-1], axis=0, dtype=int, out=send_offset[1:])
        comm.Alltoallv([recv_idx_buf, recv_cnt, recv_offset, MPI.LONG], 
                    [send_idx_buf, send_cnt, send_offset, MPI.LONG])
        out_map[f"{name}_send_idx_buf"].append(send_idx_buf)
        out_map[f"{name}_send_cnt"].append(send_cnt)
        out_map[f"{name}_send_offset"].append(send_offset)
        out_map[f"{name}_send_cnt32"].append(np.asarray(send_cnt, dtype=np.int32))
        out_map[f"{name}_send_offset32"].append(np.asarray(send_offset, dtype=np.int32))
        out_map[f"{name}_recv_idx_buf"].append(recv_idx_buf)
        out_map[f"{name}_recv_cnt"].append(recv_cnt)
        out_map[f"{name}_recv_offset"].append(recv_offset)
        out_map[f"{name}_recv_cnt32"].append(np.asarray(recv_cnt, dtype=np.int32))
        out_map[f"{name}_recv_offset32"].append(np.asarray(recv_offset, dtype=np.int32))
    
def fetch_t(indices_map, t2, kpts, kqrts, rmat, nocc, nvir, name, dtype, iphase, label='oovv', trans='nncc', init_zeros=False):
    dim = {'o': nocc, 'v': nvir}
    tensor_shape = tuple(dim[c.lower()] for c in label)
    data_size = int(np.prod(tensor_shape, dtype=np.int64))
    send_idx_buf = indices_map[f"{name}_send_idx_buf"][iphase]
    send_cnt = indices_map[f"{name}_send_cnt32"][iphase]
    send_offset = indices_map[f"{name}_send_offset32"][iphase]
    recv_idx_buf = indices_map[f"{name}_recv_idx_buf"][iphase]
    recv_cnt = indices_map[f"{name}_recv_cnt32"][iphase]
    recv_offset = indices_map[f"{name}_recv_offset32"][iphase]
    metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 'label': label,
                'trans': trans, 'incore': True, 'owned_m':recv_idx_buf}
    t2_tmp = zeros(tensor_shape, dtype=dtype, metadata=metadata)
    if init_zeros:
        return t2_tmp
    nsend = len(send_idx_buf)
    dtype = np.dtype(dtype)
    send_data = _get_work_buffer(
        indices_map, ("fetch_send", name, iphase, label, dtype.str, data_size),
        (nsend, data_size), dtype,
    )
    for i, m in enumerate(send_idx_buf):
        send_data[i, :] = np.asarray(_direct_ibz_block(t2, kqrts, m)).reshape(-1)
    block_dt = _mpi_block_dtype(dtype, data_size)
    nrecv = len(recv_idx_buf)
    recv_data = _get_work_buffer(
        indices_map, ("fetch_recv", name, iphase, label, dtype.str, data_size),
        (nrecv, data_size), dtype,
    )
    comm.Alltoallv([send_data.reshape(-1), send_cnt, send_offset, block_dt],
                   [recv_data.reshape(-1), recv_cnt, recv_offset, block_dt])
    t2_tmp.data = recv_data.reshape(nrecv, *tensor_shape)
    return t2_tmp

def sendback_t(indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, name, dtype, iphase, label='oovv'):
    dim = {'o': nocc, 'v': nvir}
    tensor_shape = tuple(dim[c.lower()] for c in label)
    data_size = int(np.prod(tensor_shape, dtype=np.int64))
    send_idx_buf = indices_map[f"{name}_send_idx_buf"][iphase]
    send_cnt = indices_map[f"{name}_send_cnt32"][iphase]
    send_offset = indices_map[f"{name}_send_offset32"][iphase]
    recv_idx_buf = indices_map[f"{name}_recv_idx_buf"][iphase]
    recv_cnt = indices_map[f"{name}_recv_cnt32"][iphase]
    recv_offset = indices_map[f"{name}_recv_offset32"][iphase]
    nrecv = len(recv_idx_buf)
    dtype = np.dtype(dtype)
    recv_data = _get_work_buffer(
        indices_map, ("sendback_send", name, iphase, label, dtype.str, data_size),
        (nrecv, data_size), dtype,
    )
    for i, m in enumerate(recv_idx_buf):
        recv_data[i, :] = np.asarray(_direct_ibz_block(t2new_tmp, kqrts, m)).reshape(-1)
    block_dt = _mpi_block_dtype(dtype, data_size)
    nsend = len(send_idx_buf)
    send_data = _get_work_buffer(
        indices_map, ("sendback_recv", name, iphase, label, dtype.str, data_size),
        (nsend, data_size), dtype,
    )
    comm.Alltoallv([recv_data.reshape(-1), recv_cnt, recv_offset, block_dt],
                   [send_data.reshape(-1), send_cnt, send_offset, block_dt])
    for i, m in enumerate(send_idx_buf):
        _add_direct_ibz_block(t2new, kqrts, m, send_data[i].reshape(tensor_shape))

def generate_my_indices(cc):
    ttt = time.perf_counter()    
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    kpts = cc.kpts
    kqrts = cc.kqrts
    kconserv = cc.khelper.kconserv
    t2_owned = cc.t2_owned
    my_indices_map = {}
    ########## For distribution
    ibz = np.asarray(kqrts.kqrts_ibz, dtype=int)
    n_ibz = ibz.shape[0]
    nk = kpts.nkpts
    owners = np.empty(n_ibz, dtype=int)
    for m,(ki,kj,ka,kb) in enumerate(ibz):
        owners[m] = m % size
    bz2ibz = np.empty((nk, nk, nk), dtype=int)
    bz2owner = np.empty((nk, nk, nk), dtype=int)
    for kk in range(nk):
        for kl in range(nk):
            for ka in range(nk):
                kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                m = kqrts.bz2ibz[kk_bz]
                bz2ibz[kk, kl, ka] = m
                bz2owner[kk, kl, ka] = owners[m]
    def _exchange_map(owner_map, idx):
        ki, kj, ka = idx
        m = int(bz2ibz[ki, kj, ka])
        r = int(bz2owner[ki, kj, ka])
        owner_map[r].add(m)
    indirect_map = build_indirect_map(kpts, kqrts)
    # t1
    need_by_owner = [{r: set() for r in range(size)}]
    for idx in t2_owned:
        ki, kk, ka, kc = kqrts.kqrts_ibz[idx]
        if ka == ki and kk == kc:
            _exchange_map(need_by_owner[0], (kk, ki, kk))
            _exchange_map(need_by_owner[0], (ki, kk, kk))
    exchange_data(need_by_owner, "t1", my_indices_map)
    # Woooo
    s2_idx = s2_index(kqrts.kqrts_ibz)
    mask = np.ones(len(kqrts.kqrts_ibz), dtype=bool)
    mask[s2_idx] = False
    not_s2 = kqrts.kqrts_ibz[mask]
    if rank == 0:
        def _Woooo(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            kk, kl, ki, kj = kq 
            total = 0
            total += int(indirect_map[kk, kl, ki]) 
            total += int(indirect_map[kl, kk, kj])
            total += int(indirect_map[kk, kl, ki])
            total += int(indirect_map[kk, kl, :].sum())
            total += int(indirect_map[ki, kj, :].sum())
            return total
        assignments, load, weights = plan_tasks(not_s2, kpts, kqrts, size*nphase, _Woooo, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["Woooo"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["Woooo"][iphase]:
            kk, kl, ki, kj = not_s2[idx]
            for ka in range(nk):
                _exchange_map(need_by_owner[iphase], (ki, kj, ka))
    exchange_data(need_by_owner, "Woooo", my_indices_map, nphase_tmp=nphase)
    # Wvoov
    if rank == 0:
        def _Wvoov(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ka, kk, ki, kc = kq
            total = 0
            total += int(indirect_map[ka, kk, ki])
            total += int(indirect_map[:,  ki, ka].sum())
            total += int(indirect_map[kk, :, kc].sum())
            total += int(indirect_map[:, kk, kc].sum())
            total += int(indirect_map[ ki, :,  ka].sum())
            return total
        assignments, load, weights, Wvoov_owners = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _Wvoov, indirect_map, nphase=nphase, return_owner=True)
    else:
        assignments = load = weights = Wvoov_owners = None
    assignments, load, weights, Wvoov_owners = comm.bcast((assignments, load, weights, Wvoov_owners), root=0)
    my_indices_map["Wvoov"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["Wvoov"][iphase]:
            ka, kk, ki, kc = kqrts.kqrts_ibz[idx]
            for kx in range(nk):
                _exchange_map(need_by_owner[iphase], (kx, ki, ka))
                _exchange_map(need_by_owner[iphase], (ki, kx, ka))
    exchange_data(need_by_owner, "Wvoov", my_indices_map, nphase_tmp=nphase)
    # Wvovo
    if rank == 0:
        def _Wvovo(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ka, kk, kc, ki = kq
            total = 0
            total += int(indirect_map[kk,ka,ki])
            total += int(indirect_map[:,kk,kc].sum())
            total += int(indirect_map[:,ki,ka].sum())
            return total
        assignments, load, weights, Wvovo_owners = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _Wvovo, indirect_map, nphase=nphase, return_owner=True)
    else:
        assignments = load = weights = Wvovo_owners = None
    assignments, load, weights, Wvovo_owners = comm.bcast((assignments, load, weights, Wvovo_owners), root=0)
    my_indices_map["Wvovo"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["Wvovo"][iphase]:
            ka, kk, kc, ki = kqrts.kqrts_ibz[idx]
            for kx in range(nk):
                _exchange_map(need_by_owner[iphase], (kx, ki, ka))
    exchange_data(need_by_owner, "Wvovo", my_indices_map, nphase_tmp=nphase)
    # Construct fast and slow ibz for t2 update
    fast_ibz = []
    slow_ibz = []
    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        if lib.isin_1d((kj,ki,kb,ka), kqrts.kqrts_ibz):
            fast_ibz.append(kq)
        else:
            slow_ibz.append(kq)
    # t2oooo fast
    if rank == 0:
        def _t2oooo_fast(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ki, kj, ka, kb = kq
            total = 0
            for kl in range(nk):
                kk = kconserv[kj, kl, ki]
                if indirect_map[kk, kl, ka]: total += 1
                if indirect_map[kk, kl, ki]: total += 1
            return total
        assignments, load, weights = plan_tasks(fast_ibz, kpts, kqrts, size*nphase, _t2oooo_fast, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2oooo_fast"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2oooo_fast"][iphase]:
            ki, kj, ka, kb = fast_ibz[idx]
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
            _exchange_map(need_by_owner_out[iphase], (kj, ki, kb))
            for kl in range(nk):
                kk = kconserv[kj, kl, ki]
                _exchange_map(need_by_owner_in[iphase], (kk, kl, ka))
    exchange_data(need_by_owner_out, "t2oooo_fast_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2oooo_fast_in", my_indices_map, nphase_tmp=nphase) 
    # t2oooo slow
    if rank == 0:
        def _t2oooo_slow(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ki, kj, ka, kb = kq
            total = 0
            for kl in range(nk):
                kk = kconserv[kj, kl, ki]
                if indirect_map[kk, kl, ka]: total += 1
                if indirect_map[kk, kl, ki]: total += 1 
                kk = kconserv[ki, kl, kj]
                if indirect_map[kk, kl, kb]: total += 1
                if indirect_map[kk, kl, kj]: total += 1
            return total
        assignments, load, weights = plan_tasks(slow_ibz, kpts, kqrts, size*nphase, _t2oooo_slow, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2oooo_slow"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2oooo_slow"][iphase]:
            ki, kj, ka, kb = slow_ibz[idx]
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
            for kl in range(nk):
                kk = kconserv[kj, kl, ki]
                _exchange_map(need_by_owner_in[iphase], (kk, kl, ka))
                kk = kconserv[ki, kl, kj]
                _exchange_map(need_by_owner_in[iphase], (kk, kl, kb))
    exchange_data(need_by_owner_out, "t2oooo_slow_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2oooo_slow_in", my_indices_map, nphase_tmp=nphase) 
    # t2voov1 fast
    if rank == 0:
        def _t2voov1_fast(kq, kpts, kqrts, indirect_map):
            ki, kj, ka, kb = kq
            total = int(indirect_map[ki, kj, ka]) + int(indirect_map[kj, ki, kb])
            return total
        assignments, load, weights = plan_tasks(fast_ibz, kpts, kqrts, size*nphase, _t2voov1_fast, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2voov1_fast"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov1_fast"][iphase]:
            ki, kj, ka, kb = fast_ibz[idx]
            _exchange_map(need_by_owner_in[iphase], (ki, kj, ka))
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
            _exchange_map(need_by_owner_out[iphase], (kj, ki, kb))
    exchange_data(need_by_owner_out, "t2voov1_fast_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2voov1_fast_in", my_indices_map, nphase_tmp=nphase) 
    # t2voov1 slow
    if rank == 0:
        def _t2voov1_slow(kq, kpts, kqrts, indirect_map):
            ki, kj, ka, kb = kq
            total = int(indirect_map[ki, kj, ka]) + int(indirect_map[kj, ki, kb])
            total += int(indirect_map[kj, ki, kb]) + int(indirect_map[ki, kj, ka])
            return total
        assignments, load, weights = plan_tasks(slow_ibz, kpts, kqrts, size*nphase, _t2voov1_slow, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2voov1_slow"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov1_slow"][iphase]:
            ki, kj, ka, kb = slow_ibz[idx]
            _exchange_map(need_by_owner_in[iphase], (ki, kj, ka))
            _exchange_map(need_by_owner_in[iphase], (kj, ki, kb))
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
    exchange_data(need_by_owner_out, "t2voov1_slow_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2voov1_slow_in", my_indices_map, nphase_tmp=nphase)
    # t2voov2 fast
    if rank == 0:
        def _t2voov2_fast(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ki, kj, ka, kb = kq
            def core(ki_, kj_, ka_, kb_):
                cnt = 0
                for kk in range(nk):
                    kc = kconserv[ka_, ki_, kk]
                    if indirect_map[ka_, kk, ki_]: cnt += 1
                    if indirect_map[ka_, kk, kc]: cnt += 1
                    if indirect_map[kk, kj_, kc]: cnt += 1 
                    if indirect_map[kk, kj_, kb_]: cnt += 1  
                    kc1 = kconserv[kk, ka_, kj_]
                    if indirect_map[kb_, kk, kc1]: cnt += 1
                    if indirect_map[kk, kj_, ka_]: cnt += 1
                return cnt
            total = core(ki, kj, ka, kb)
            return total
        assignments, load, weights = plan_tasks(fast_ibz, kpts, kqrts, size*nphase, _t2voov2_fast, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2voov2_fast"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov2_fast"][iphase]:
            ki, kj, ka, kb = fast_ibz[idx]
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
            _exchange_map(need_by_owner_out[iphase], (kj, ki, kb))
            for kk in range(nk):
                kc = kconserv[ka, ki, kk]
                _exchange_map(need_by_owner_in[iphase], (kk, kj, kc))
                _exchange_map(need_by_owner_in[iphase], (kk, kj, kb))
                kc = kconserv[kk, ka, kj]
                _exchange_map(need_by_owner_in[iphase], (kk, kj, ka))
    exchange_data(need_by_owner_out, "t2voov2_fast_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2voov2_fast_in", my_indices_map, nphase_tmp=nphase) 
    # Wvoov owner map
    Wvoov_bz2ibz = np.empty((nk, nk, nk), dtype=int)
    Wvoov_bz2owner = np.empty((nk, nk, nk), dtype=int)
    for kk in range(nk):
        for kl in range(nk):
            for ka in range(nk):
                kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                m = kqrts.bz2ibz[kk_bz]
                Wvoov_bz2ibz[kk, kl, ka] = m
                Wvoov_bz2owner[kk, kl, ka] = Wvoov_owners[m]
    def _exchange_map_Wvoov(owner_map, idx):
        ki, kj, ka = idx
        m = int(Wvoov_bz2ibz[ki, kj, ka])
        r = int(Wvoov_bz2owner[ki, kj, ka])
        owner_map[r].add(m)
    # Wvovo owner map
    Wvovo_bz2ibz = np.empty((nk, nk, nk), dtype=int)
    Wvovo_bz2owner = np.empty((nk, nk, nk), dtype=int)
    for kk in range(nk):
        for kl in range(nk):
            for ka in range(nk):
                kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                m = kqrts.bz2ibz[kk_bz]
                Wvovo_bz2ibz[kk, kl, ka] = m
                Wvovo_bz2owner[kk, kl, ka] = Wvovo_owners[m]
    def _exchange_map_Wvovo(owner_map, idx):
        ki, kj, ka = idx
        m = int(Wvovo_bz2ibz[ki, kj, ka])
        r = int(Wvovo_bz2owner[ki, kj, ka])
        owner_map[r].add(m)
    # fetch Wvoov and Wvovo
    need_by_owner_Wvoov = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_Wvovo = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov2_fast"][iphase]:
            ki, kj, ka, kb = fast_ibz[idx]
            for kk in range(nk):
                kc = kconserv[ka, ki, kk]
                _exchange_map_Wvoov(need_by_owner_Wvoov[iphase], (ka, kk, ki))
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (ka, kk, kc))
                kc = kconserv[kk, ka, kj]
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (kb, kk, kc))
    exchange_data(need_by_owner_Wvoov, "Wvoov_fast", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_Wvovo, "Wvovo_fast", my_indices_map, nphase_tmp=nphase) 
    # t2voov2 slow
    if rank == 0:
        def _t2voov2_slow(kq, kpts, kqrts, indirect_map):
            nk = kpts.nkpts
            ki, kj, ka, kb = kq
            def core(ki_, kj_, ka_, kb_):
                cnt = 0
                for kk in range(nk):
                    kc = kconserv[ka_, ki_, kk]
                    if indirect_map[ka_, kk, ki_]: cnt += 1
                    if indirect_map[ka_, kk, kc]: cnt += 1
                    if indirect_map[kk, kj_, kc]: cnt += 1 
                    if indirect_map[kk, kj_, kb_]: cnt += 1  
                    kc1 = kconserv[kk, ka_, kj_]
                    if indirect_map[kb_, kk, kc1]: cnt += 1
                    if indirect_map[kk, kj_, ka_]: cnt += 1
                return cnt
            total = core(ki, kj, ka, kb)
            total += core(kj, ki, kb, ka)
            return total
        assignments, load, weights = plan_tasks(slow_ibz, kpts, kqrts, size*nphase, _t2voov2_slow, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2voov2_slow"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov2_slow"][iphase]:
            ki, kj, ka, kb = slow_ibz[idx]
            _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
            for kk in range(nk):
                kc = kconserv[ka, ki, kk]
                _exchange_map(need_by_owner_in[iphase], (kk, kj, kc))
                _exchange_map(need_by_owner_in[iphase], (kk, kj, kb))
                kc = kconserv[kk, ka, kj]
                _exchange_map(need_by_owner_in[iphase], (kk, kj, ka))
                ## kj,ki,kb,ka
                kc = kconserv[kb, kj, kk]
                _exchange_map(need_by_owner_in[iphase], (kk, ki, kc))
                _exchange_map(need_by_owner_in[iphase], (kk, ki, ka))
                kc = kconserv[kk, kb, ki]
                _exchange_map(need_by_owner_in[iphase], (kk, ki, kb))
    exchange_data(need_by_owner_out, "t2voov2_slow_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2voov2_slow_in", my_indices_map, nphase_tmp=nphase) 
    # fetch Wvoov and Wvovo
    need_by_owner_Wvoov = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_Wvovo = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2voov2_slow"][iphase]:
            ki, kj, ka, kb = slow_ibz[idx]
            for kk in range(nk):
                kc = kconserv[ka, ki, kk]
                _exchange_map_Wvoov(need_by_owner_Wvoov[iphase], (ka, kk, ki))
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (ka, kk, kc))
                kc = kconserv[kk, ka, kj]
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (kb, kk, kc))
                ## kj,ki,kb,ka
                kc = kconserv[kb, kj, kk]
                _exchange_map_Wvoov(need_by_owner_Wvoov[iphase], (kb, kk, kj))
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (kb, kk, kc))
                kc = kconserv[kk, kb, ki]
                _exchange_map_Wvovo(need_by_owner_Wvovo[iphase], (ka, kk, kc))
    exchange_data(need_by_owner_Wvoov, "Wvoov_slow", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_Wvovo, "Wvovo_slow", my_indices_map, nphase_tmp=nphase) 
    # t2vvvv
    kakb, igroup = np.unique(kqrts.kqrts_ibz[:,2:], axis=0, return_inverse=True)
    igroup = igroup.ravel()
    group_members = [np.where(igroup==i)[0] for i in range(kakb.shape[0])]
    units = np.array([(i, kc) for i in range(kakb.shape[0]) for kc in range(kpts.nkpts)], dtype=int)
    if rank == 0:
        r = 2.0 * nocc / (nvir * nvir) + 2.0 / nvir
        def _t2vvvv(unit, kpts, kqrts, indirect_map):
            i, kc = unit
            n_members = len(group_members[i])
            n_indirect = 0
            for m in group_members[i]:
                ki, kj, _, _ = kqrts.kqrts_ibz[m]
                n_indirect += int(indirect_map[ki, kj, kc])
            return n_members + r * n_indirect
        assignments, load, weights = plan_tasks(units, kpts, kqrts, size*nphase, _t2vvvv, indirect_map, nphase=nphase)
    else:
        assignments = load = weights = None
    assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
    my_indices_map["t2vvvv"] = assignments[rank*nphase: (rank+1)*nphase]
    need_by_owner_out = [{r: set() for r in range(size)} for _ in range(nphase)]
    need_by_owner_in = [{r: set() for r in range(size)} for _ in range(nphase)]
    for iphase in range(nphase):
        for idx in my_indices_map["t2vvvv"][iphase]:
            i, kc = units[idx]
            ka, kb = kakb[i]
            kd = kconserv[ka, kc, kb]
            for x in group_members[i]:
                ki, kj, _, _ = kqrts.kqrts_ibz[x]
                _exchange_map(need_by_owner_out[iphase], (ki, kj, ka))
                _exchange_map(need_by_owner_in[iphase], (ki, kj, kc))
    exchange_data(need_by_owner_out, "t2vvvv_out", my_indices_map, nphase_tmp=nphase)    
    exchange_data(need_by_owner_in, "t2vvvv_in", my_indices_map, nphase_tmp=nphase) 
    print('rank =', rank, get_mem(),  'Generating indices', time.perf_counter() - ttt)
    cc.my_indices_map, cc.fast_ibz, cc.slow_ibz = my_indices_map, fast_ibz, slow_ibz
    
##############################################################
##############################################################
# Periodic CCSD
def cc_Foo(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    #Fki = np.empty((nkpts,nocc,nocc), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'oo', 'trans': 'cn',
                'incore': True}
    Fki = zeros([nocc,nocc], dtype=t2.dtype, metadata=metadata)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kl, kc, kd = kq
        kk = ki
        Soovv = 2 * eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
        fock = einsum('klcd,ilcd->ki', Soovv, t2[ki,kl,kc])
        if ki == kc:
            fock += einsum('klcd,ic,ld->ki', Soovv, t1[ki], t1[kl])
        for _, iop in kqrts.loop_stabilizer(i):
            rmat_oo = rmat.oo[ki][iop]
            Fki[ki] += einsum('ki,km,in->mn', fock, rmat_oo.conj(), rmat_oo)
    return Fki

def cc_Fov(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    #Fkc = np.empty((nkpts,nocc,nvir), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'ov', 'trans': 'cn',
                'incore': True}
    Fkc = zeros([nocc,nvir], dtype=t2.dtype, metadata=metadata)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        kk, kl, kc, kd = kq
        if kc == kk and kl == kd:
            Soovv = 2 * eris.oovv[kk,kl,kk] - eris.oovv[kk,kl,kl].transpose(0,1,3,2)
            fock = einsum('klcd,ld->kc', Soovv, t1[kl])
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[kk][iop]
                rmat_vv = rmat.vv[kk][iop]
                Fkc[kk] += einsum('kc,km,cb->mb', fock, rmat_oo.conj(), rmat_vv)
    return Fkc

def cc_Fvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    #Fac = np.empty((nkpts,nvir,nvir), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'vv', 'trans': 'cn',
                'incore': True}
    Fac = zeros([nvir,nvir], dtype=t2.dtype, metadata=metadata)
    ka_ibz_bz = kpts.ibz2bz[np.arange(kpts.nkpts_ibz)]
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        kk, kl, ka, kd = kq
        kc = ka
        Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
        fock = -einsum('klcd,klad->ac', Soovv, t2[kk,kl,ka])
        if kk == ka:
            fock += -einsum('klcd,ka,ld->ac', Soovv, t1[ka], t1[kl])
        op_group = kqrts.stars_ops[i]
        ka_prim = kpts.k2opk[ka, op_group]
        mask = np.isin(ka_prim, ka_ibz_bz)
        for iop, ka_p in zip(op_group[mask], ka_prim[mask]):
            rmat_vv = rmat.vv[ka][iop]
            Fac[ka_p] += einsum('ac,ae,cf->ef', fock, rmat_vv.conj(), rmat_vv)
    return Fac

def cc_Loo(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape

    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'oo', 'trans': 'cn',
                'incore': True}
    Lki = zeros([nocc,nocc], dtype=t2.dtype, metadata=metadata)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kl, ka, kb = kq
        if ki == ka:
            fock = (2*einsum('klic,lc->ki', eris.ooov[ki,kl,ki], t1[kl])
                     -einsum('lkic,lc->ki', eris.ooov[kl,ki,ki], t1[kl]))
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[ki][iop]
                Lki[ki] += einsum('ki,km,in->mn', fock, rmat_oo.conj(), rmat_oo)
    return Lki

def cc_Lvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'vv', 'trans': 'cn',
                'incore': True}
    Lac = zeros([nvir,nvir], dtype=t2.dtype, metadata=metadata)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ka, kk, kc, kl = kq
        if ka == kc:
            Svovv = 2 * eris.vovv[ka,kk,ka] - eris.vovv[ka,kk,kk].transpose(0,1,3,2)
            fock = einsum('akcd,kd->ac', Svovv, t1[kk])
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_vv = rmat.vv[ka][iop]
                Lac[ka] += einsum('ac,ae,cf->ef', fock, rmat_vv.conj(), rmat_vv)
    return Lac

def cc_Woooo(Wklij, not_s2, t1, t2, eris, my_indices):
    nkpts, nocc, nvir = t1.shape
    for i in my_indices:
        kq = not_s2[i]
        kk, kl, ki, kj = kq
        oooo  = einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
        oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
        oooo += eris.oooo[kk,kl,ki]
        vvoo = eris.oovv[kk,kl].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
        t2t  = t2[ki,kj].copy().transpose(0,3,4,1,2)
        t2t[ki] += einsum('ic,jd->cdij',t1[ki],t1[kj])
        t2t = t2t.reshape(nkpts*nvir,nvir,nocc,nocc)
        oooo += einsum('cdkl,cdij->klij',vvoo,t2t)
        Wklij[kk,kl,ki] = oooo
    return Wklij

def cc_Wvoov(Wakic, kqrts, t1, t2, eris, my_indices):
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        ka, kk, ki, kc = kq
        voov  = einsum('akdc,id->akic', eris.vovv[ka,kk,ki], t1[ki])
        voov -= einsum('lkic,la->akic', eris.ooov[ka,kk,ki], t1[ka])
        voov += eris.voov[ka,kk,ki]
        kd = ki
        tau = t2[:,ki,ka].copy()
        tau[ka] += 2*einsum('id,la->liad', t1[kd], t1[ka])
        oovv_tmp = np.array(eris.oovv[kk,:,kc])
        voov -= 0.5*einsum('xklcd,xliad->akic', oovv_tmp, tau)
        Soovv_tmp = 2*oovv_tmp - eris.oovv[:,kk,kc].transpose(0,2,1,3,4)
        voov += 0.5*einsum('xklcd,xilad->akic', Soovv_tmp, t2[ki,:,ka])
        Wakic[ka,kk,ki] = voov
    return Wakic

def cc_Wvovo(Wakci, kqrts, t1, t2, eris, my_indices):
    nkpts, nocc, nvir = t1.shape
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        ka, kk, kc, ki = kq
        vovo  = einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
        vovo -= einsum('klic,la->akci',eris.ooov[kk,ka,ki],t1[ka])
        vovo += np.asarray(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)
        oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
        t2f   = t2[:,ki,ka].copy()
        kd = ki
        t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
        t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
        vovo -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
        Wakci[ka,kk,kc] = vovo
    return Wakci

def get_diff_norm_2(old, new):
    old = old.ravel()
    new = new.ravel()
    r, blksize = 0.0, 24 * 1024 * 1024
    for i in range(0, len(old), blksize):
        r += np.linalg.norm(new[i:i+blksize] - old[i:i+blksize]) ** 2
    return r

def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None, callback=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]
    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)
    if isinstance(mycc.diis, MPIDIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = MPIDIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
        adiis.t1_size = mycc.kpts.nkpts_ibz * mycc.nocc * (mycc.nmo - mycc.nocc)
    else:
        adiis = None
    generate_my_indices(mycc)
    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        if callback is not None:
            callback(locals())
        normt1_2 = get_diff_norm_2(t1new.data, t1.data)
        normt2_2 = get_diff_norm_2(t2new.data, t2.data)
        normt2_2 = comm.allreduce(normt2_2, op=MPI.SUM)
        normt = (normt1_2 + normt2_2) ** 0.5
        if mycc.iterative_damping < 1.0:
            alpha = np.asarray(mycc.iterative_damping)
            if isinstance(t1, tuple):
                t1new = tuple((1-alpha) * np.asarray(t1_part) + alpha * np.asarray(t1new_part)
                    for t1_part, t1new_part in zip(t1, t1new))
                t2new = tuple((1-alpha) * np.asarray(t2_part) + alpha * np.asarray(t2new_part)
                    for t2_part, t2new_part in zip(t2, t2new))
            else:
                t1new = (1-alpha) * np.asarray(t1) + alpha * np.asarray(t1new)
                t2new *= alpha
                t2new += (1-alpha) * np.asarray(t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        comm.Bcast(t1.data, root=0)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        mycc.cycles = istep + 1
        log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, name, eccsd, eccsd - eold, normt)
        cput1 = log.timer(f'{name} iter', *cput1)
        report_mem('CC iter')
        if mycc.save_dir is not None and mycc.save_per_iter:
            np.save(os.path.join(mycc.save_dir, f't1_{rank}.npy'), t1.data)
            np.save(os.path.join(mycc.save_dir, f't2_{rank}.npy'), t2.data)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    if mycc.save_dir is not None and not mycc.save_per_iter:
        np.save(os.path.join(mycc.save_dir, f't1_{rank}.npy'), t1.data)
        np.save(os.path.join(mycc.save_dir, f't2_{rank}.npy'), t2.data)
    return converged, eccsd, t1, t2

def update_amps(cc, t1, t2, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    kpts = cc.kpts
    kqrts = cc.kqrts
    rmat = cc.rmat
    kconserv = cc.khelper.kconserv
    fast_ibz = cc.fast_ibz
    slow_ibz = cc.slow_ibz
    my_indices_map = cc.my_indices_map
    t2_owned = cc.t2_owned
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
    ki_ibz_bz = kpts.ibz2bz[np.arange(kpts.nkpts_ibz)]
    fov = fock[:, :nocc, nocc:]
    kconserv = cc.khelper.kconserv
    ttt = time.perf_counter()
    ##############################################################
    ##############################################################
    Foo = cc_Foo(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    comm.Allreduce(MPI.IN_PLACE, Foo.data, op=MPI.SUM)
    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Foo[ki] += eris.fock[ki,:nocc,:nocc]
    Fov = cc_Fov(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    comm.Allreduce(MPI.IN_PLACE, Fov.data, op=MPI.SUM)
    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Fov[ki] += eris.fock[ki,:nocc,nocc:]
    Fvv = cc_Fvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    comm.Allreduce(MPI.IN_PLACE, Fvv.data, op=MPI.SUM)
    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Fvv[ki] += eris.fock[ki,nocc:,nocc:]
    Loo = cc_Loo(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    comm.Allreduce(MPI.IN_PLACE, Loo.data, op=MPI.SUM)
    Loo.data += Foo.data
    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        Loo[ki] += einsum('kc,ic->ki', eris.fock[:,:nocc,nocc:][ki], t1[ki])
    Lvv = cc_Lvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    comm.Allreduce(MPI.IN_PLACE, Lvv.data, op=MPI.SUM)
    Lvv.data += Fvv.data
    for ka_ibz in range(kpts.nkpts_ibz):
        ka = kpts.ibz2bz[ka_ibz]
        Lvv[ka] -= einsum('kc,ka->ac', eris.fock[:,:nocc,nocc:][ka], t1[ka])
    local_dt = time.perf_counter() - ttt
    Fov = Fov.todense()
    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        Foo[ki][np.diag_indices(nocc)] -= mo_e_o[ki]
        Fvv[ki][np.diag_indices(nvir)] -= mo_e_v[ki]
        Loo[ki][np.diag_indices(nocc)] -= mo_e_o[ki]
        Lvv[ki][np.diag_indices(nvir)] -= mo_e_v[ki]
    ##############################################################
    ##############################################################
    t1new = zeros_like(t1)
    t1 = t1.todense()
    t2new = zeros_like(t2)
    local_dt = time.perf_counter() - ttt
    # T1 equation
    nibz = kpts.nkpts_ibz
    base = nibz // size
    rem = nibz % size
    xstart = rank * base + min(rank, rem)
    xend = xstart + base + (1 if rank < rem else 0)
    for ka_ibz in range(xstart, xend):
        ki = ka = kpts.ibz2bz[ka_ibz]
        t1new[ka]  = fov[ka].conj()
        t1new[ka] += -2. * einsum('kc,ka,ic->ia', fov[ki], t1[ka], t1[ki])
        t1new[ka] += einsum('ac,ic->ia', Fvv[ka], t1[ki])
        t1new[ka] += -einsum('ki,ka->ia', Foo[ki], t1[ka])
    ##############################################################
    ##############################################################
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kk, kc, kd = kq
        ka = ki
        Svovv = 2 * eris.vovv[ka, kk, kc] - eris.vovv[ka, kk, kd].transpose(0, 1, 3, 2)
        tau_term_1 = t2[ki, kk, kc].copy()
        if ki == kc and kk == kd:
            tau_term_1 += einsum('ic,kd->ikcd', t1[ki], t1[kk])
        fock = einsum('akcd,ikcd->ia', Svovv, tau_term_1)
        for _, iop in kqrts.loop_stabilizer(i):
            rmat_oo = rmat.oo[ka][iop]
            rmat_vv = rmat.vv[ka][iop]
            t1new[ka] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())
    ##############################################################
    ##############################################################
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        kk, kl, ki, kc = kq
        ka = ki
        Sooov = 2 * eris.ooov[kk, kl, ki] - eris.ooov[kl, kk, ki].transpose(1, 0, 2, 3)
        tau_term_1 = t2[kk, kl, ka].copy()
        if kk == ka and kl == kc:
            tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])
        fock = -einsum('klic,klac->ia', Sooov, tau_term_1)
        
        op_group = kqrts.stars_ops[i]
        ka_prim = kpts.k2opk[ka, op_group]
        mask = np.isin(ka_prim, ki_ibz_bz)
        for iop, ka_p in zip(op_group[mask], ka_prim[mask]):
            rmat_oo = rmat.oo[ka][iop]
            rmat_vv = rmat.vv[ka][iop]
            t1new[ka_p] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())
    ##############################################################
    ##############################################################
    t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t1", eris.fock.dtype, iphase=0)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kk, ka, kc = kq
        if ka == ki and kk == kc:
            tau_term = 2 * t2_tmp[kk, ki, kk] - t2_tmp[ki, kk, kk].transpose(1, 0, 2, 3)
            if ki == kk:
                tau_term += einsum('ic,ka->kica', t1[ki], t1[ka])

            fock = einsum('kc,kica->ia', Fov[kc], tau_term)
            fock += einsum('akic,kc->ia', 2 * eris.voov[ka, kk, ki], t1[kc])
            fock += einsum('kaic,kc->ia', -eris.ovov[kk, ka, ki], t1[kc])

            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[ka][iop]
                rmat_vv = rmat.vv[ka][iop]
                t1new[ka] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())
    t2_tmp = None
    comm.Allreduce(MPI.IN_PLACE, t1new.data, op=MPI.SUM)
    for ki_ibz in range(kpts.nkpts_ibz):
        ka = ki = kpts.ibz2bz[ki_ibz]
        # Remove zero/padded elements from denominator
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        t1new[ki] /= eia
    ##############################################################
    ##############################################################
    # T2 equation
    Loo = Loo.todense()
    Lvv = Lvv.todense()
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kj, ka, kb = kq
        t2new[ki, kj, ka] = eris.oovv[ki, kj, ka].conj()
    ##############################################################
    ##############################################################
    s2_idx = s2_index(kqrts.kqrts_ibz)
    mask = np.ones(len(kqrts.kqrts_ibz), dtype=bool)
    mask[s2_idx] = False
    not_s2 = kqrts.kqrts_ibz[mask]
    metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                'label': 'oooo', 'trans': 'ccnn', 'incore': True}
    Woooo = zeros([nocc,nocc,nocc,nocc], dtype=t1.dtype, metadata=metadata)
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "Woooo", eris.fock.dtype, iphase=iphase)
        cc_Woooo(Woooo, not_s2, t1, t2_tmp, eris, my_indices_map["Woooo"][iphase])
        t2_tmp = None
    comm.Allreduce(MPI.IN_PLACE, Woooo.data, op=MPI.SUM)
    s2_idx = s2_index(kqrts.kqrts_ibz)
    for i, kq in enumerate(kqrts.kqrts_ibz[s2_idx]):
        kl, kk, kj, ki = kq
        Woooo[kl,kk,kj] = Woooo[kk,kl,ki].transpose(1,0,3,2)
    ##############################################################
    ##############################################################
    t2_tmp = None
    t2new_tmp = None
    def _t2_oooo(ki,kj,ka,kb):
        t2_tmp_new = 0
        for kl in range(nkpts):
            kk = kconserv[kj, kl, ki]
            tau_term = t2_tmp[kk, kl, ka].copy()
            if kl == kb and kk == ka:
                tau_term += einsum('ic,jd->ijcd', t1[ka], t1[kb])
            t2_tmp_new += 0.5 * einsum('klij,klab->ijab', Woooo[kk, kl, ki], tau_term)
        return t2_tmp_new
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2oooo_fast_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2oooo_fast_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        for i in my_indices_map["t2oooo_fast"][iphase]:
            kq = fast_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_oooo(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2new_tmp[kj, ki, kb] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2oooo_fast_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2oooo_slow_in", eris.fock.dtype, iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2oooo_slow_out", eris.fock.dtype, iphase, init_zeros=True)
        for i in my_indices_map["t2oooo_slow"][iphase]:
            kq = slow_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_oooo(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2_tmp_new = _t2_oooo(kj,ki,kb,ka)
            t2new_tmp[ki, kj, ka] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2oooo_slow_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    Woooo = None
    ##############################################################
    ##############################################################
    _add_vvvv(cc, t2new, t1, t2, eris, my_indices_map)
    ##############################################################
    ##############################################################
    def _t2_voov1(ki,kj,ka,kb):
        t2ija = t2_tmp[ki, kj, ka].copy()
        t2_tmp_new = einsum('ac,ijcb->ijab', Lvv[ka], t2ija)
        t2_tmp_new += einsum('ki,kjab->ijab', -Loo[ki], t2ija)
        del t2ija
        kc = kj
        tmp2 = np.asarray(eris.vovv[kc, ki, kb]).transpose(3, 2, 1, 0).conj() \
               - einsum('kbic,ka->abic', eris.ovov[ka, kb, ki], t1[ka])
        t2_tmp_new += einsum('abic,jc->ijab', tmp2, t1[kj])
        kk = kb
        tmp2 = np.asarray(eris.ooov[kj, ki, kk]).transpose(3, 2, 1, 0).conj() \
               + einsum('akic,jc->akij', eris.voov[ka, kk, ki], t1[kj])
        t2_tmp_new -= einsum('akij,kb->ijab', tmp2, t1[kb])
        return t2_tmp_new
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2voov1_fast_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2voov1_fast_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        for i in my_indices_map["t2voov1_fast"][iphase]:
            kq = fast_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_voov1(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2new_tmp[kj, ki, kb] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2voov1_fast_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2voov1_slow_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2voov1_slow_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        for i in my_indices_map["t2voov1_slow"][iphase]:
            kq = slow_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_voov1(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2_tmp_new = _t2_voov1(kj,ki,kb,ka)
            t2new_tmp[ki, kj, ka] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2voov1_slow_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    ##############################################################
    ##############################################################
    Wvoov_total_indices = [x for i in my_indices_map["Wvoov"] for x in i]
    metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 'label': 'voov', 
                'trans': 'ccnn', 'incore': True, 'owned_m': Wvoov_total_indices}
    Wvoov = zeros([nvir,nocc,nocc,nvir], dtype=t1.dtype, metadata=metadata)
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "Wvoov", eris.fock.dtype, iphase=iphase)
        cc_Wvoov(Wvoov, kqrts, t1, t2_tmp, eris, my_indices_map["Wvoov"][iphase])
        t2_tmp = None
    ##############################################################
    ##############################################################
    Wvovo_total_indices = [x for i in my_indices_map["Wvovo"] for x in i]
    metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 'label': 'vovo', 
                'trans': 'ccnn', 'incore': True, 'owned_m': Wvovo_total_indices}
    Wvovo = zeros([nvir,nocc,nvir,nocc], dtype=t1.dtype, metadata=metadata)
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "Wvovo", eris.fock.dtype, iphase=iphase)
        cc_Wvovo(Wvovo, kqrts, t1, t2_tmp, eris, my_indices_map["Wvovo"][iphase])
        t2_tmp = None
    ##############################################################
    ##############################################################
    Wvoov_tmp = None
    Wvovo_tmp = None
    def _t2_voov2(ki,kj,ka,kb):
        t2_tmp_new = 0
        for kk in range(nkpts):
            kc = kconserv[ka, ki, kk]
            tmp_Wvoov = Wvoov_tmp[ka, kk, ki].copy()
            tmp_voov = 2. * tmp_Wvoov - Wvovo_tmp[ka, kk, kc].transpose(0, 1, 3, 2)
            t2_tmp_new += einsum('akic,kjcb->ijab', tmp_voov, t2_tmp[kk, kj, kc])
            t2_tmp_new -= einsum('akic,kjbc->ijab', tmp_Wvoov, t2_tmp[kk, kj, kb])
            kc = kconserv[kk, ka, kj]
            t2_tmp_new -= einsum('bkci,kjac->ijab', Wvovo_tmp[kb, kk, kc], t2_tmp[kk, kj, ka])
            tmp_Wvoov = None
        return t2_tmp_new
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2voov2_fast_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2voov2_fast_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        Wvoov_tmp = fetch_t(my_indices_map, Wvoov, kpts, kqrts, rmat, nocc, nvir, "Wvoov_fast", eris.fock.dtype, iphase=iphase, label='voov', trans='ccnn', init_zeros=False)
        Wvovo_tmp = fetch_t(my_indices_map, Wvovo, kpts, kqrts, rmat, nocc, nvir, "Wvovo_fast", eris.fock.dtype, iphase=iphase, label='vovo', trans='ccnn', init_zeros=False)
        for i in my_indices_map["t2voov2_fast"][iphase]:
            kq = fast_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_voov2(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2new_tmp[kj, ki, kb] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = Wvoov_tmp = Wvovo_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2voov2_fast_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2voov2_slow_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, t2new, kpts, kqrts, rmat, nocc, nvir, "t2voov2_slow_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        Wvoov_tmp = fetch_t(my_indices_map, Wvoov, kpts, kqrts, rmat, nocc, nvir, "Wvoov_slow", eris.fock.dtype, iphase=iphase, label='voov', trans='ccnn', init_zeros=False)
        Wvovo_tmp = fetch_t(my_indices_map, Wvovo, kpts, kqrts, rmat, nocc, nvir, "Wvovo_slow", eris.fock.dtype, iphase=iphase, label='vovo', trans='ccnn', init_zeros=False)
        for i in my_indices_map["t2voov2_slow"][iphase]:
            kq = slow_ibz[i]
            ki, kj, ka, kb = kq
            t2_tmp_new = _t2_voov2(ki,kj,ka,kb)
            t2new_tmp[ki, kj, ka] += t2_tmp_new
            t2_tmp_new = _t2_voov2(kj,ki,kb,ka)
            t2new_tmp[ki, kj, ka] += t2_tmp_new.transpose(1, 0, 3, 2)
        t2_tmp = Wvoov_tmp = Wvovo_tmp = None
        sendback_t(my_indices_map, t2new_tmp, t2new, kqrts, nocc, nvir, "t2voov2_slow_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    Wvovo = Wvoov = None
    ##############################################################
    ##############################################################
    for i in t2_owned:
        ki, kj, ka, kb = kqrts.kqrts_ibz[i]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]
        t2new[ki, kj, ka] /= eijab
    
    logger.timer_debug1(cc, 'update_amps', *time0)
    
    return t1new, t2new


def _add_vvvv(cc, Ht2, t1, t2, eris, my_indices_map):
    kpts = cc.kpts
    kqrts = cc.kqrts
    rmat = cc.rmat
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = kpts.nkpts
    kconserv = cc.khelper.kconserv
    def _get_Wvvvv(ka, kb, kc):
        Lpq = eris.Lpq
        kd = kconserv[ka, kc, kb]
        Lbd = (Lpq[kb,kd][:,nocc:,nocc:] -
                einsum('Lkd,kb->Lbd', Lpq[kb,kd][:,:nocc,nocc:], t1[kb]))
        Wvvvv = einsum('Lac,Lbd->abcd', Lpq[ka,kc][:,nocc:,nocc:], Lbd)
        Lbd = None
        kcbd = einsum('Lkc,Lbd->kcbd', Lpq[ka,kc][:,:nocc,nocc:],
                            Lpq[kb,kd][:,nocc:,nocc:])
        Wvvvv -= einsum('kcbd,ka->abcd', kcbd, t1[ka])
        Wvvvv *= (1. / nkpts)
        return Wvvvv
    kakb, igroup = np.unique(kqrts.kqrts_ibz[:,2:], axis=0, return_inverse=True)
    igroup = igroup.ravel()
    group_members = [np.where(igroup==i)[0] for i in range(kakb.shape[0])]
    units = np.array([(i, kc) for i in range(kakb.shape[0]) for kc in range(kpts.nkpts)], dtype=int)
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "t2vvvv_in", eris.fock.dtype, iphase=iphase, init_zeros=False)
        t2new_tmp = fetch_t(my_indices_map, Ht2, kpts, kqrts, rmat, nocc, nvir, "t2vvvv_out", eris.fock.dtype, iphase=iphase, init_zeros=True)
        for x in my_indices_map["t2vvvv"][iphase]:
            i, kc = units[x]
            ka, kb = kakb[i]
            kd = kconserv[ka, kc, kb]
            Wvvvv = _get_Wvvvv(ka, kb, kc)
            for m in group_members[i]:
                ki, kj, _, _ = kqrts.kqrts_ibz[m]
                tau = t2_tmp[ki, kj, kc].copy()
                if ki == kc and kj == kd:
                    tau += einsum('ic,jd->ijcd', t1[ki], t1[kj])
                t2new_tmp[ki, kj, ka] += einsum('abcd,ijcd->ijab', Wvvvv, tau)
        t2_tmp = None
        sendback_t(my_indices_map, t2new_tmp, Ht2, kqrts, nocc, nvir, "t2vvvv_out", eris.fock.dtype, iphase)
        t2new_tmp = None
    return Ht2


def energy(cc, t1, t2, eris):
    kpts = cc.kpts
    kqrts = cc.kqrts
    t2_owned = cc.t2_owned
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    e_tol = 0.0
    e = 0.0
    
    nibz = kpts.nkpts_ibz
    base = nibz // size
    rem = nibz % size
    xstart = rank * base + min(rank, rem)
    xend = xstart + base + (1 if rank < rem else 0)
    for ki_ibz in range(xstart, xend):
        ki = kpts.ibz2bz[ki_ibz]
        weight = kpts.weights_ibz[ki_ibz]
        e += 2 * einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki]) * weight
    
    tau = zeros_like(t2)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kj, ka, kb = kq
        tau[ki, kj, ka] = t2[ki, kj, ka]
        if ki == ka and kj == kb:
            tau[ki, kj, ka] += einsum('ia,jb->ijab', t1[ki], t1[kj])
            
    kq_weights = kqrts.weights_ibz
    for k in t2_owned:
        kq = kqrts.kqrts_ibz[k]
        ki, kj, ka, kb = kq
        weight = kq_weights[k] * nkpts**3
        e += 2 * einsum('ijab,ijab', tau[ki, kj, ka], eris.oovv[ki, kj, ka]) * weight
        e -= einsum('ijab,ijba', tau[ki, kj, ka], eris.oovv[ki, kj, kb]) * weight
    e_tol = comm.allreduce(e, op=MPI.SUM)

    e_tol /= nkpts
    if abs(e_tol.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e_tol)
    return e_tol.real

def _update_procs_mf(mf):
    '''Update mean-field objects to be the same on all processors'''
    mf1 = mf.copy()

    mo_coeff  = comm.bcast(mf.mo_coeff, root=0)
    mo_energy = comm.bcast(mf.mo_energy, root=0)
    mo_occ    = comm.bcast(mf.mo_occ, root=0)
    kpts      = comm.bcast(mf.kpts, root=0)
    e_tot     = comm.bcast(mf.e_tot, root=0)

    mf1.mo_coeff = mo_coeff
    mf1.mo_energy = mo_energy
    mf1.mo_occ = mo_occ
    mf1.kpts  = kpts
    mf1.e_tot  = e_tot
    mf1.converged = True
    comm.Barrier()
    return mf1


class compute_block:
    def __init__(self, funct):
        self._compute = funct

    def __getitem__(self, key):
        k = tuple(key)
        arr = self._compute(*k)
        return arr


class RCCSD(pyscf.pbc.cc.kccsd_rhf.RCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, save_dir=None, save_per_iter=False):
        '''
        Attributes:
            ktensor_direct : bool
                If set to True, the tensors will be stored as block-sparse,
                and the symmetry related blocks are computed on-the-fly when needed.
                Otherwise, the tensors will be converted to dense tensors whenever
                there is enough memory. Default is False.
            eris_outcore : bool
                If set to True, the integrals will be always stored on the disk.
                Otherwise, whether the integrals are stored on the disk or in memory
                depends on the available memory size. Default is False.
        '''
        mf = _update_procs_mf(mf)
        pyscf.pbc.cc.kccsd_rhf.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kqrts = KQuartets(mf.kpts).build()
        self.rmat = None
        self.ktensor_direct = False
        self.eris_outcore = False
        self.t2_incore = True
        self.save_dir = save_dir
        self.eom_imds = None
        self.save_per_iter = save_per_iter
        t2_tuples = []
        t2_owned = []
        for m, (ki,kj,ka,kb) in enumerate(self.kqrts.kqrts_ibz):
            owner = m % size
            if owner == rank:
                t2_tuples.append((ki,kj,ka))
                t2_owned.append(m)
        self.t2_owned = t2_owned
        self.t2_tuples = t2_tuples
    
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        ttt = time.perf_counter()
        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None or (not self.incore):
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        if mbpt2:
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
            return self.e_corr, self.t1, self.t2

        self.converged, self.e_corr, self.t1, self.t2 = \
            kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                   tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                   verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2
    
    def make_eom_imds(self, partition=None, eris=None):
        imds = _IMDS(self)
        imds.make_ip_ea(partition=partition)
        return imds
    
    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, kptlist=None):
        
        if self.eom_imds is None:
            self.eom_imds = self.make_eom_imds(partition=partition)
        
        eomip = EOMIP(self)
        if rank != 0:
            eomip.verbose = 0
        
        return eomip.kernel(nroots=nroots, left=left,
                                    koopmans=koopmans, guess=guess,
                                    partition=partition, eris=None,
                                    imds=self.eom_imds, kptlist=kptlist)
    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, kptlist=None):
        
        if self.eom_imds is None:
            self.eom_imds = self.make_eom_imds(partition=partition)
            
        eomea = EOMEA(self)
        if rank != 0:
            eomea.verbose = 0
        
        return eomea.kernel(nroots=nroots, left=left,
                                    koopmans=koopmans, guess=guess,
                                    partition=partition, eris=None,
                                    imds=self.eom_imds, kptlist=kptlist)

    def ao2mo(self, mo_coeff=None):
        ttt = time.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        eris = _DFERIs()
        eris._common_init_(self, mo_coeff)
        # use padded mo_coeff to construct the rotation matrix
        self.rmat = MORotationMatrix(self.kpts, eris.mo_coeff, self._scf.get_ovlp(), eris.nocc, eris.nmo)
        self.rmat.build()
        _init_df_eris(self, eris)
        kpts = self.kpts.kpts
        nkpts = len(kpts)
        kqrts = self.kqrts
        nocc = eris.nocc
        nvir = eris.nvir
        khelper = self.khelper
        kconserv = khelper.kconserv
        mo_coeff = eris.mo_coeff
        dtype = eris.dtype
        t2_owned = self.t2_owned
        
        def _compute_integral(ki, kj, ka, block=None):
            kb = kconserv[ki, ka, kj]
            if block == 'oooo':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, :nocc, :nocc])
            elif block == 'ooov':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, :nocc, nocc:])
            elif block == 'oovv':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, nocc:], eris.Lpq[kj, kb][:, :nocc, nocc:])
            elif block == 'ovov':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, nocc:, nocc:])
            elif block == 'voov':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, nocc:, :nocc], eris.Lpq[kj, kb][:, :nocc, nocc:])
            elif block == 'vovv':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, nocc:, nocc:], eris.Lpq[kj, kb][:, :nocc, nocc:])
            elif block == 'vvvv':
                phys_eri = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, nocc:, nocc:], eris.Lpq[kj, kb][:, nocc:, nocc:])
            else:
                raise ValueError(f"Unknown block type {block}")
        
            return phys_eri / nkpts
        
        # Save small integrals
        cput1 = logger.process_clock(), logger.perf_counter()
        kptlist = kqrts.kqrts_ibz[:,:3][:,[0,2,1]] #chemists' notation
        khelper.build_symm_map(kptlist=kptlist)
        common_metadata = {'kpts'  : self.kpts,
                           'kqrts' : self.kqrts,
                           'rmat'  : self.rmat,
                           'trans' : 'ccnn',
                           'incore': True}
        eris.oooo = zeros([nocc,nocc,nocc,nocc], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oooo'})
        eris.ooov = zeros([nocc,nocc,nocc,nvir], dtype=dtype,
                                metadata={**common_metadata, 'label': 'ooov'})
        eris.oovv = zeros([nocc,nocc,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oovv'})
        eris.oooo.data *= 0.0
        eris.ooov.data *= 0.0
        eris.oovv.data *= 0.0
        # eris.ovov = zeros([nocc,nvir,nocc,nvir], dtype=dtype,
        #                         metadata={**common_metadata, 'label': 'ovov'})
        # eris.voov = zeros([nvir,nocc,nocc,nvir], dtype=dtype,
        #                         metadata={**common_metadata, 'label': 'voov'})
        for i in t2_owned:
            kijab = kqrts.kqrts_ibz[i]
            ki, kj, ka, kb = kijab
            eris.oooo[ki, kj, ka] = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, :nocc, :nocc]) / nkpts
            eris.ooov[ki, kj, ka] = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, :nocc, nocc:]) / nkpts
            eris.oovv[ki, kj, ka] = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, nocc:], eris.Lpq[kj, kb][:, :nocc, nocc:]) / nkpts
            # eris.ovov[ki, kj, ka] = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, :nocc, :nocc], eris.Lpq[kj, kb][:, nocc:, nocc:]) / nkpts
            # eris.voov[ki, kj, ka] = einsum('Lia, Ljb->ijab', eris.Lpq[ki, ka][:, nocc:, :nocc], eris.Lpq[kj, kb][:, :nocc, nocc:]) / nkpts
        comm.Allreduce(MPI.IN_PLACE, eris.oooo.data, op=MPI.SUM)
        safeAllreduceInPlace_ksymm(comm, eris.ooov)
        safeAllreduceInPlace_ksymm(comm, eris.oovv)
        cput1 = log.timer_debug1('computing oooo, ooov, oovv', *cput1)
        
        # eris.oooo = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='oooo'))
        # eris.ooov = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='ooov'))
        # eris.oovv = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='oovv'))
        eris.ovov = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='ovov'))
        eris.voov = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='voov'))
        eris.vovv = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='vovv'))
        eris.vvvv = compute_block(funct=lambda ki, kj, ka: _compute_integral(ki, kj, ka, block='vvvv'))
        return eris
        
    def init_amps(self, eris):
        time0 = logger.process_clock(), logger.perf_counter()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kpts = self.kpts
        kqrts = self.kqrts
        rmat = self.rmat
        assert rmat is not None
        t2_owned = self.t2_owned

        metadata = {'kpts': kpts, 'rmat': rmat,
                    'label': 'ov', 'trans': 'nc', 'incore': True, 'prefix':'t1_init'}
        t1 = zeros((nocc, nvir), dtype=eris.fock.dtype, metadata=metadata)
        
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 'label': 'oovv', 'trans': 'nncc',
                    'incore': self.t2_incore, 'prefix':'t2_init', 'owned_m':t2_owned}
        t2 = zeros((nocc,nocc,nvir,nvir), dtype=eris.fock.dtype, metadata=metadata)
        mo_e_o = [eris.mo_energy[k][:nocc] for k in range(nkpts)]
        mo_e_v = [eris.mo_energy[k][nocc:] for k in range(nkpts)]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        emp2 = 0.0
        local_mp2 = 0.0
        for i in t2_owned:
            ki, kj, ka, kb = kqrts.kqrts_ibz[i]
            weight = kqrts.weights_ibz[i] * nkpts**3
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ka,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                        [0,nvir,kb,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]
            eris_ijab = eris.oovv[ki, kj, ka]
            eris_ijba = eris.oovv[ki, kj, kb]
            t2[ki, kj, ka] = eris_ijab.conj() / eijab
            woovv = 2 * eris_ijab - eris_ijba.transpose(0, 1, 3, 2)
            local_energy = einsum('ijab,ijab', t2[ki, kj, ka], woovv) * weight
            local_mp2 += local_energy
            
        emp2 = comm.allreduce(local_mp2, op=MPI.SUM)
        self.emp2 = emp2.real
        self.emp2 /= nkpts
        
        if rank == 0:
            logger.info(self, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
            logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2
    
    def amplitudes_to_vector(self, t1, t2):
        t1_raw = np.asarray(getattr(t1, 'data', t1))
        t2_raw = np.asarray(getattr(t2, 'data', t2))
        return np.concatenate((t1_raw, t2_raw), axis=None)

    def vector_to_amplitudes(self, vec):
        kpts = self.kpts
        kqrts = self.kqrts
        rmat = self.rmat
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1_size = kpts.nkpts_ibz * nocc * nvir
        t1_flat = vec[:t1_size]
        t2_flat = vec[t1_size:]

        metadata = {'kpts': kpts, 'rmat': rmat,
                    'label': 'ov', 'trans': 'nc', 'incore': True, 'prefix':'t1_v2amp'}
        t1 = fromraw(t1_flat, (nocc,nvir), dtype=vec.dtype,
                             metadata=metadata)

        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'oovv', 'trans': 'nncc', 'incore': self.t2_incore, 'owned_m':self.t2_owned}
        t2 = fromraw(t2_flat, (nocc,nocc,nvir,nvir), dtype=vec.dtype,
                             metadata=metadata)
        return t1, t2
    
    energy = energy
    update_amps = update_amps

def _init_df_eris(cc, eris):
    from pyscf.ao2mo import _ao2mo
    if cc._scf.with_df._cderi is None:
        cc._scf.with_df.build()

    cell = cc._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    kpts = getattr(cc.kpts, 'kpts', cc.kpts)
    nkpts = len(kpts)
    #naux = cc._scf.with_df.get_naoaux()
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *eris.mo_coeff)
    eris.Lpq = Lpq = np.empty((nkpts,nkpts), dtype=object)

    tao = []
    ao_loc = None
#    with df.CDERIArray(cc._scf.with_df._cderi) as cderi_array:
#        for ki in range(nkpts):
#            for kj in range(nkpts):
#                Lpq = cderi_array[ki,kj]
    for ki, kpti in enumerate(kpts):
        for kj, kptj in enumerate(kpts):
            kpti_kptj = np.array((kpti, kptj))
            # This loader is compatible with the old GDF format
            with df._load3c(cc._scf.with_df._cderi, 'j3c', kpti_kptj) as j3c:
                Lpq_tmp = np.asarray(j3c)

                mo = np.hstack((eris.mo_coeff[ki], eris.mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_tmp, mo, (0, nmo, nmo, nmo + nmo), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_tmp[0].size != nao**2: # aosym = 's2'
                        Lpq_tmp = lib.unpack_tril(Lpq_tmp).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_tmp, mo, (0, nmo, nmo, nmo + nmo), tao, ao_loc)
                Lpq[ki,kj] = out.reshape(-1,nmo,nmo)
    cc._scf.with_df._cderi = None
    return eris

class _DFERIs:
    def __init__(self, cell=None):
        self.kpts = None
        self.mo_coeff = None
        self.nocc = None
        self.nmo = None
        self.nvir = None
        self.fock = None
        self.dtype = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovov = None
        self.voov = None
        self.vovv = None
        self.vvvv = None
        self.Lpq = None

    def _common_init_(self, cc, mo_coeff=None):
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        mf = cc._scf
        cell = mf.cell
        self.kpts = kpts = cc.kpts
        self.nocc = nocc = cc.nocc
        self.nmo = nmo = cc.nmo
        self.nvir = nmo - nocc

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        self.dtype = mo_coeff[-1].dtype
        # Re-make our fock MO matrix elements from density and fock AO
        # FIXME what if mo_coeff is not consistent with cc.mo_occ?
        dm = mf.make_rdm1(mo_coeff, cc.mo_occ)
        exxdiv = mf.exxdiv if cc.keep_exxdiv else None
        with lib.temporary_env(mf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            vhf = mf.get_veff(cell, dm)
        fockao = mf.get_hcore() + vhf
        self.mo_coeff = mo_coeff = padded_mo_coeff(cc, mo_coeff)
        fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo)) for k, mo in enumerate(mo_coeff)])
        self.fock = fock
        mo_energy = [fock[k].diagonal().real for k in range(len(fock))]
        if not cc.keep_exxdiv:
            # Add HFX correction in the self.mo_energy to improve convergence in
            # CCSD iteration. It is useful for the 2D systems since their occupied and
            # the virtual orbital energies may overlap which may lead to numerical
            # issue in the CCSD iterations.
            # FIXME: Whether to add this correction for other exxdiv treatments?
            # Without the correction, MP2 energy may be largely off the correct value.
            madelung = tools.madelung(cell, kpts.kpts)
            mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                         for k, mo_e in enumerate(mo_energy)]
        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")
        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [mo_energy[kp][nonzero_padding[kp]] for kp in range(len(mo_energy))]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)
        self.mo_energy = mo_energy
        
########################################
# EOM-IP-CCSD
########################################
def generate_eom_indices(my_indices_map, imds, kshift=None):
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv
    kpts = imds.kpts
    kqrts = imds.kqrts
    ibz = np.asarray(kqrts.kqrts_ibz, dtype=int)
    n_ibz = ibz.shape[0]
    owners = np.empty(n_ibz, dtype=int)
    for m,(ki,kj,ka,kb) in enumerate(ibz):
        owners[m] = m % size
    bz2ibz = np.empty((nkpts, nkpts, nkpts), dtype=int)
    bz2owner = np.empty((nkpts, nkpts, nkpts), dtype=int)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ka in range(nkpts):
                kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                m = kqrts.bz2ibz[kk_bz]
                bz2ibz[kk, kl, ka] = m
                bz2owner[kk, kl, ka] = owners[m]
    def _exchange_map(owner_map, idx):
        ki, kj, ka = idx
        m = int(bz2ibz[ki, kj, ka])
        r = int(bz2owner[ki, kj, ka])
        owner_map[r].add(m)
    indirect_map = build_indirect_map(kpts, kqrts)
    def num_indirect_Wvvvv(ka, kb, kc):
        total = 0
        kd = kconserv[ka, kc, kb]
        for kk in range(nkpts):
            kl = kconserv[kc,kk,kd]
            total += int(indirect_map[kk,kl,kc])
            total += int(indirect_map[kk,kl,ka])
        return total
    def num_indirect_Wvvvo(ka,kb,kc):
        total = 0
        kj = kconserv[ka,kc,kb]
        total += int(indirect_map[kb,ka,kj])
        total += int(indirect_map[ka,kb,kc])
        for kl in range(nkpts):
            kd = kconserv[ka,kc,kl]
            total += int(indirect_map[kl,kj,kd])
            total += int(indirect_map[kl,kj,kb])
            total += int(indirect_map[kl,kj,kd])
            kd = kconserv[kb,kc,kl]
            total += int(indirect_map[kj,kl,kd])
            kk = kconserv[kb,kl,ka]
            total += int(indirect_map[kl,kk,kj])
            total += int(indirect_map[kl,kk,kb])
        total += int(indirect_map[kb,ka,kj])
        total += int(indirect_map[kc,kj,ka])
        total += num_indirect_Wvvvv(ka,kb,kc)
        return total
    if kshift is not None:
        ##############################################################
        ##############################################################
        # From imds
        # W1ovov owner map
        W1ovov_owners = my_indices_map["W1ovov_owners"]
        W1ovvo_owners = my_indices_map["W1ovvo_owners"]
        W1ovov_bz2ibz = np.empty((nkpts, nkpts, nkpts), dtype=int)
        W1ovov_bz2owner = np.empty((nkpts, nkpts, nkpts), dtype=int)
        for kk in range(nkpts):
            for kl in range(nkpts):
                for ka in range(nkpts):
                    kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                    m = kqrts.bz2ibz[kk_bz]
                    W1ovov_bz2ibz[kk, kl, ka] = m
                    W1ovov_bz2owner[kk, kl, ka] = W1ovov_owners[m]
        def _exchange_map_W1ovov(owner_map, idx):
            ki, kj, ka = idx
            m = int(W1ovov_bz2ibz[ki, kj, ka])
            r = int(W1ovov_bz2owner[ki, kj, ka])
            owner_map[r].add(m)
        # W1ovvo owner map
        W1ovvo_bz2ibz = np.empty((nkpts, nkpts, nkpts), dtype=int)
        W1ovvo_bz2owner = np.empty((nkpts, nkpts, nkpts), dtype=int)
        for kk in range(nkpts):
            for kl in range(nkpts):
                for ka in range(nkpts):
                    kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                    m = kqrts.bz2ibz[kk_bz]
                    W1ovvo_bz2ibz[kk, kl, ka] = m
                    W1ovvo_bz2owner[kk, kl, ka] = W1ovvo_owners[m]
        def _exchange_map_W1ovvo(owner_map, idx):
            ki, kj, ka = idx
            m = int(W1ovvo_bz2ibz[ki, kj, ka])
            r = int(W1ovvo_bz2owner[ki, kj, ka])
            owner_map[r].add(m)
        ##############################################################
        ##############################################################
        # ipccsd_matvec
        ktuples = []
        for ki in range(nkpts): 
            for kj in range(nkpts):
                ktuples.append((ki, kj, kshift))
        if rank == 0:        
            def _ipccsd_matvec(kq, kpts, kqrts, indirect_map):
                total = 0
                ki, kj, kshift = kq
                kb = kconserv[ki, kshift, kj]
                for kl in range(nkpts):
                    kk = kconserv[ki, kl, kj]
                    total += int(indirect_map[kk, kl, ki])
                    kd = kconserv[kl, kj, kb]
                    total += int(indirect_map[kl, kb, kd])
                    total += int(indirect_map[kl, kb, kd])
                    total += int(indirect_map[kl, kb, kj])
                    kd = kconserv[kl, ki, kb]
                    total += int(indirect_map[kl, kb, ki])
                total += int(indirect_map[ki, kj, kshift])
                total += int(indirect_map[kj, ki, kshift])
                total += int(indirect_map[ki, kj, kshift])
                return total
            assignments, load, weights = plan_tasks(ktuples, kpts, kqrts, size*nphase, _ipccsd_matvec, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["ipccsd_matvec"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovvo = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["ipccsd_matvec"][iphase]:
                ki, kj, kshift = ktuples[idx]
                _exchange_map(need_by_owner[iphase], (ki, kj, kshift))
                kb = kconserv[ki, kshift, kj]
                for kl in range(nkpts):
                    kd = kconserv[kl, kj, kb]
                    _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kl, kb, kd))
                    _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kl, kb, kd))
                    _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kl, kb, kj))
                    kd = kconserv[kl, ki, kb]
                    _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kl, kb, ki))
        exchange_data(need_by_owner, "ipccsd_matvec", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovov, "ipccsd_matvec_Wovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovvo, "ipccsd_matvec_Wovvo", my_indices_map, nphase_tmp=nphase)
        ##############################################################
        ##############################################################
        # ipccsd_diag
        if rank == 0:
            def _ipccsd_diag(kq, kpts, kqrts, indirect_map):
                total = 0
                ki, kj, kshift = kq
                kb = kconserv[ki, kshift, kj]
                if ki == kconserv[ki, kj, kj]:
                    total += int(indirect_map[ki, kj, ki])
                total += int(indirect_map[kj, kb, kj])
                total += int(indirect_map[kj, kb, kb])
                total += int(indirect_map[ki, kb, ki])
                kd = kconserv[kj, kshift, ki]
                total += int(indirect_map[ki, kj, kshift])
                total += int(indirect_map[kj, ki, kd])
                total += int(indirect_map[ki, kj, kshift])
                total += int(indirect_map[ki, kj, kd])
                return total
            assignments, load, weights = plan_tasks(ktuples, kpts, kqrts, size*nphase, _ipccsd_diag, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["ipccsd_diag"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovvo = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["ipccsd_diag"][iphase]:
                ki, kj, kshift = ktuples[idx]
                kb = kconserv[ki, kshift, kj]
                _exchange_map(need_by_owner[iphase], (ki, kj, kshift))
                _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kj, kb, kj))
                _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (ki, kb, ki))
                _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kj, kb, kb))
        exchange_data(need_by_owner, "ipccsd_diag", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovov, "ipccsd_diag_Wovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovvo, "ipccsd_diag_Wovvo", my_indices_map, nphase_tmp=nphase)
        ##############################################################
        ##############################################################
        # eaccsd_matvec
        # r2vvvv
        if rank == 0:
            def _r2vvvv(kq, kpts, kqrts, indirect_map):
                ka, kb, kc, kd = kq
                n_indirect = num_indirect_Wvvvv(ka, kb, kc)
                kk_bz = kpts.ktuple_to_index((ka,kb,kc))
                kk_ibz = kqrts.bz2ibz[kk_bz]
                assert (kqrts.kqrts_ibz[kk_ibz] == (ka,kb,kc,kd)).all()
                op_group = kqrts.stars_ops[kk_ibz]
                n_bz = len(op_group)
                return (n_bz+n_indirect)
            assignments, load, weights = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _r2vvvv, indirect_map, nphase=nphase)    
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["r2vvvv"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["r2vvvv"][iphase]:
                ka, kb, kc, kd = kqrts.kqrts_ibz[idx]
                for kk in range(nkpts):
                    kl = kconserv[kc,kk,kd]
                    _exchange_map(need_by_owner[iphase], (kk,kl,ka))
        exchange_data(need_by_owner, "r2vvvv", my_indices_map, nphase_tmp=nphase)
        ##############################################################
        ##############################################################
        # r2vvvo
        ktuples = []
        for ki in range(nkpts): 
            for kj in range(nkpts):
                ktuples.append((ki, kj, kshift))
        if rank == 0:
            def _r2vvvo(kq, kpts, kqrts, indirect_map):
                total = 0
                kj, ka, kshift = kq
                kb = kconserv[kshift,ka,kj]
                total += num_indirect_Wvvvo(ka,kb,kshift)
                return total
            assignments, load, weights = plan_tasks(ktuples, kpts, kqrts, size*nphase, _r2vvvo, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["r2vvvo"] = assignments[rank*nphase: (rank+1)*nphase]
        # For Wvvvo
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_W1ovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_W1ovvo = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["r2vvvo"][iphase]:
                kj, ka, kc = ktuples[idx]
                kb = kconserv[kc,ka,kj]
                # Wvvvo ka,kb,kc
                kj = kconserv[ka,kc,kb]
                _exchange_map_W1ovov(need_by_owner_W1ovov[iphase], (kb,ka,kj))
                _exchange_map_W1ovvo(need_by_owner_W1ovvo[iphase], (ka,kb,kc))
                for kl in range(nkpts):
                    kd = kconserv[ka,kc,kl]
                    _exchange_map(need_by_owner[iphase], (kl,kj,kd))
                    _exchange_map(need_by_owner[iphase], (kl,kj,kb))
                    _exchange_map(need_by_owner[iphase], (kl,kj,kd))
                    kd = kconserv[kb,kc,kl]
                    _exchange_map(need_by_owner[iphase], (kj,kl,kd))
                    kk = kconserv[kb,kl,ka]
                    _exchange_map(need_by_owner[iphase], (kl,kk,kb))
                _exchange_map(need_by_owner[iphase], (kc,kj,ka))  
                kj = kconserv[ka,kc,kb]
                kd = kconserv[ka,kc,kb]
                for kk in range(nkpts):
                    kl = kconserv[kc,kk,kd]
                    _exchange_map(need_by_owner[iphase], (kk,kl,ka))
        exchange_data(need_by_owner, "eom_Wvvvo", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_W1ovov, "eom_Wvvvo_W1ovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_W1ovvo, "eom_Wvvvo_W1ovvo", my_indices_map, nphase_tmp=nphase)  
        ##############################################################
        ##############################################################
        # eaccsd_matvec
        ktuples = []
        for ki in range(nkpts): 
            for kj in range(nkpts):
                ktuples.append((ki, kj, kshift))
        if rank == 0:                    
            def _eaccsd_matvec(kq, kpts, kqrts, indirect_map):
                total = 0
                kj, ka, kshift = kq
                kb = kconserv[kshift, ka, kj]
                for kd in range(nkpts):
                    kl = kconserv[kd, kb, kj]
                    total += int(indirect_map[kl, kb, kd])
                    total += int(indirect_map[kl, kb, kj])
                    total += int(indirect_map[kl, kb, kd])
                    kl = kconserv[kd, ka, kj]
                    total += int(indirect_map[kl, ka, kj])
                return total
            assignments, load, weights = plan_tasks(ktuples, kpts, kqrts, size*nphase, _eaccsd_matvec, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["eaccsd_matvec"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovvo = [{r: set() for r in range(size)} for _ in range(nphase)]        
        for iphase in range(nphase):
            for idx in my_indices_map["eaccsd_matvec"][iphase]:
                kj, ka, kshift = ktuples[idx]
                kb = kconserv[kshift, ka, kj]
                for kd in range(nkpts):
                    kl = kconserv[kd, kb, kj]
                    _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kl, kb, kd))
                    _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kl, kb, kj))
                    _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kl, kb, kd))
                    kl = kconserv[kd, ka, kj]
                    _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kl, ka, kj))
                _exchange_map(need_by_owner[iphase], (kshift, kj, ka))
        exchange_data(need_by_owner, "eaccsd_matvec", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovov, "eaccsd_matvec_Wovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovvo, "eaccsd_matvec_Wovvo", my_indices_map, nphase_tmp=nphase) 
        ##############################################################
        ##############################################################
        # eaccsd_diag
        ktuples = []
        for ki in range(nkpts): 
            for kj in range(nkpts):
                ktuples.append((ki, kj, kshift))
        if rank == 0:
            def _eaccsd_diag(kq, kpts, kqrts, indirect_map):
                total = 0
                kj, ka, kshift = kq
                kb = kconserv[kshift, ka, kj]
                total += num_indirect_Wvvvv(ka, kb, ka)
                total += int(indirect_map[kj, kb, kj])
                total += int(indirect_map[kj, kb, kb])
                total += int(indirect_map[kj, ka, kj])
                total += int(indirect_map[kshift, kj, ka])
                total += int(indirect_map[kshift, kj, ka])
                total += int(indirect_map[kshift, kj, ka])
                total += int(indirect_map[kshift, kj, kb])
                return total
            assignments, load, weights = plan_tasks(ktuples, kpts, kqrts, size*nphase, _eaccsd_diag, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["eaccsd_diag"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_Wovvo = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["eaccsd_diag"][iphase]:
                kj, ka, kshift = ktuples[idx]
                kb = kconserv[kshift, ka, kj]
                # Wvvvv (ka,kb,ka)
                kdd = kconserv[ka, ka, kb]
                for kk in range(nkpts):
                    kl = kconserv[ka,kk,kdd]
                    _exchange_map(need_by_owner[iphase], (kk,kl,ka))
                # Done
                _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kj, kb, kj))
                _exchange_map_W1ovvo(need_by_owner_Wovvo[iphase], (kj, kb, kb))
                _exchange_map_W1ovov(need_by_owner_Wovov[iphase], (kj, ka, kj))
                _exchange_map(need_by_owner[iphase], (kshift, kj, ka))
        exchange_data(need_by_owner, "eaccsd_diag", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovov, "eaccsd_diag_Wovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_Wovvo, "eaccsd_diag_Wovvo", my_indices_map, nphase_tmp=nphase)
        return my_indices_map
        
    else:
        # EOM Woooo 
        if rank == 0:
            def _eom_Woooo(kq, kpts, kqrts, indirect_map):
                kk, kl, ki, kj = kq
                total = 0
                total += int(indirect_map[kk,kl,ki])
                total += int(indirect_map[kk,kl,ki])
                total += int(indirect_map[kl,kk,kj])
                total += int(indirect_map[kk,kl,ki])
                for kc in range(nkpts):
                    total += int(indirect_map[kk,kl,kc])
                    total += int(indirect_map[ki,kj,kc])
                return total
            assignments, load, weights = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _eom_Woooo, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["eom_Woooo"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["eom_Woooo"][iphase]:
                kk, kl, ki, kj = kqrts.kqrts_ibz[idx]
                for kc in range(nkpts):
                    _exchange_map(need_by_owner[iphase], (ki,kj,kc))
        exchange_data(need_by_owner, "eom_Woooo", my_indices_map, nphase_tmp=nphase)
        
        # EOM Wooov
        if rank == 0:
            def _eom_Wooov(kq, kpts, kqrts, indirect_map):
                nkpts = kpts.nkpts
                kk, kl, ki, kd = kq
                total = 0
                total += int(indirect_map[kk,kl,ki])
                total += int(indirect_map[kk,kl,ki])
                return total
            assignments, load, weights = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size, _eom_Wooov, indirect_map)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["eom_Wooov"] = assignments[rank]
        
        # EOM W1ovov
        if rank == 0:
            def _eom_W1ovov(kq, kpts, kqrts, indirect_map):
                nkpts = kpts.nkpts
                kk, kb, ki, kd = kq
                total = 0
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    total += int(indirect_map[kk,kl,kc])
                    total += int(indirect_map[ki,kl,kc])
                return total
            assignments, load, weights, W1ovov_owners = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _eom_W1ovov, indirect_map, nphase=nphase, return_owner=True)
        else:
            assignments = load = weights = W1ovov_owners = None
        assignments, load, weights, W1ovov_owners = comm.bcast((assignments, load, weights, W1ovov_owners), root=0)
        my_indices_map["eom_W1ovov"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["eom_W1ovov"][iphase]:
                kk, kb, ki, kd = kqrts.kqrts_ibz[idx]
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    _exchange_map(need_by_owner[iphase], (ki,kl,kc))
        exchange_data(need_by_owner, "eom_W1ovov", my_indices_map, nphase_tmp=nphase)
        my_indices_map["W1ovov_owners"] = W1ovov_owners
        # EOM W1ovvo
        if rank == 0:
            def _eom_W1ovvo(kq, kpts, kqrts, indirect_map):
                nkpts = kpts.nkpts
                kk, ka, kc, ki = kq
                total = 0
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    total += int(indirect_map[ki,kl,ka])
                    total += int(indirect_map[kl,ki,ka])
                    total += int(indirect_map[kk,kl,kc])
                    total += int(indirect_map[kk,kl,kd])
                    total += int(indirect_map[ki,kl,ka])
                return total
            assignments, load, weights, W1ovvo_owners = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _eom_W1ovvo, indirect_map, nphase=nphase, return_owner=True)
        else:
            assignments = load = weights = W1ovvo_owners = None
        assignments, load, weights, W1ovvo_owners = comm.bcast((assignments, load, weights, W1ovvo_owners), root=0)
        my_indices_map["eom_W1ovvo"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["eom_W1ovvo"][iphase]:
                kk, ka, kc, ki = kqrts.kqrts_ibz[idx]
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    _exchange_map(need_by_owner[iphase], (ki,kl,ka))
                    _exchange_map(need_by_owner[iphase], (kl,ki,ka))
                    _exchange_map(need_by_owner[iphase], (ki,kl,ka))
        exchange_data(need_by_owner, "eom_W1ovvo", my_indices_map, nphase_tmp=nphase)
        my_indices_map["W1ovvo_owners"] = W1ovvo_owners
        # W1ovov owner map
        W1ovov_bz2ibz = np.empty((nkpts, nkpts, nkpts), dtype=int)
        W1ovov_bz2owner = np.empty((nkpts, nkpts, nkpts), dtype=int)
        for kk in range(nkpts):
            for kl in range(nkpts):
                for ka in range(nkpts):
                    kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                    m = kqrts.bz2ibz[kk_bz]
                    W1ovov_bz2ibz[kk, kl, ka] = m
                    W1ovov_bz2owner[kk, kl, ka] = W1ovov_owners[m]
        def _exchange_map_W1ovov(owner_map, idx):
            ki, kj, ka = idx
            m = int(W1ovov_bz2ibz[ki, kj, ka])
            r = int(W1ovov_bz2owner[ki, kj, ka])
            owner_map[r].add(m)
        # W1ovvo owner map
        W1ovvo_bz2ibz = np.empty((nkpts, nkpts, nkpts), dtype=int)
        W1ovvo_bz2owner = np.empty((nkpts, nkpts, nkpts), dtype=int)
        for kk in range(nkpts):
            for kl in range(nkpts):
                for ka in range(nkpts):
                    kk_bz = kpts.ktuple_to_index((kk, kl, ka))
                    m = kqrts.bz2ibz[kk_bz]
                    W1ovvo_bz2ibz[kk, kl, ka] = m
                    W1ovvo_bz2owner[kk, kl, ka] = W1ovvo_owners[m]
        def _exchange_map_W1ovvo(owner_map, idx):
            ki, kj, ka = idx
            m = int(W1ovvo_bz2ibz[ki, kj, ka])
            r = int(W1ovvo_bz2owner[ki, kj, ka])
            owner_map[r].add(m)
        # EOM Wovoo
        if rank == 0:
            def _eom_Wovoo(kq, kpts, kqrts, indirect_map):
                nkpts = kpts.nkpts
                kk, kb, ki, kj = kq
                total = 0
                total += int(indirect_map[kk,kb,ki])
                total += int(indirect_map[kk,kb,ki])
                total += int(indirect_map[kk,kb,ki])
                total += int(indirect_map[ki,kj,kk])
                for kd in range(nkpts):
                    kl = kconserv[ki,kk,kd]
                    total += int(indirect_map[kl,kj,kd])
                    total += int(indirect_map[kj,kl,kd])
                    total += int(indirect_map[kk,kl,ki])
                    total += int(indirect_map[kl,kk,ki])
                    total += int(indirect_map[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    total += int(indirect_map[kl,kk,kj])
                    total += int(indirect_map[kl,ki,kb])
                    total += int(indirect_map[kj,ki,kd])
                total += int(indirect_map[ki,kj,kk])
                return total
            assignments, load, weights = plan_tasks(kqrts.kqrts_ibz, kpts, kqrts, size*nphase, _eom_Wovoo, indirect_map, nphase=nphase)
        else:
            assignments = load = weights = None
        assignments, load, weights = comm.bcast((assignments, load, weights), root=0)
        my_indices_map["eom_Wovoo"] = assignments[rank*nphase: (rank+1)*nphase]
        need_by_owner = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_W1ovov = [{r: set() for r in range(size)} for _ in range(nphase)]
        need_by_owner_W1ovvo = [{r: set() for r in range(size)} for _ in range(nphase)]
        for iphase in range(nphase):
            for idx in my_indices_map["eom_Wovoo"][iphase]:
                kk, kb, ki, kj = kqrts.kqrts_ibz[idx]
                _exchange_map_W1ovov(need_by_owner_W1ovov[iphase], (kk,kb,ki))
                _exchange_map_W1ovvo(need_by_owner_W1ovvo[iphase], (kk,kb,ki))
                for kd in range(nkpts):
                    kl = kconserv[ki,kk,kd]
                    _exchange_map(need_by_owner[iphase], (kl,kj,kd))
                    _exchange_map(need_by_owner[iphase], (kj,kl,kd))
                    _exchange_map(need_by_owner[iphase], (kl,kj,kd))
                    kl = kconserv[kb,ki,kd]
                    _exchange_map(need_by_owner[iphase], (kl,ki,kb))
                    _exchange_map(need_by_owner[iphase], (kj,ki,kd))
                _exchange_map(need_by_owner[iphase], (ki,kj,kk))        
        exchange_data(need_by_owner, "eom_Wovoo", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_W1ovov, "eom_Wovoo_W1ovov", my_indices_map, nphase_tmp=nphase)
        exchange_data(need_by_owner_W1ovvo, "eom_Wovoo_W1ovvo", my_indices_map, nphase_tmp=nphase)
        
        return my_indices_map


def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled.'''
    if imds is None: imds = eom.make_imds()
    nmo = eom.nmo
    t2 = imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv
    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)
    kpts = imds.kpts
    kqrts = imds.kqrts
    rmat = imds.rmat
    comm.Bcast(r1, root=0)
    comm.Bcast(r2, root=0)
    my_indices_map = eom.my_indices_map
    ktuples = []
    for ki in range(nkpts): 
        for kj in range(nkpts):
            ktuples.append((ki, kj, kshift))
    Hr1 = np.zeros(r1.shape, dtype=np.result_type(imds.Loo.dtype, r1.dtype))
    base = nkpts // size
    rem = nkpts % size
    xstart = rank * base + min(rank, rem)
    xend = xstart + base + (1 if rank < rem else 0)
    # 1h-2h1p block
    for kl in range(xstart, xend):
        Hr1 += 2. * einsum('ld,ild->i', imds.Fov[kl], r2[kshift, kl])
        Hr1 += -einsum('ld,lid->i', imds.Fov[kl], r2[kl, kshift])
        for kk in range(nkpts):
            kd = kconserv[kk, kshift, kl]
            Hr1 += -2. * einsum('klid,kld->i', imds.Wooov[kk, kl, kshift], r2[kk, kl])
            Hr1 += einsum('lkid,kld->i', imds.Wooov[kl, kk, kshift], r2[kk, kl])
    comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
    if rank == 0:
        Hr1 -= einsum('ki,k->i', imds.Loo[kshift], r1)
    comm.Bcast(Hr1, root=0)
    ##############################################################
    ##############################################################
    Hr2 = np.zeros(r2.shape, dtype=np.result_type(imds.Wovoo.dtype, r1.dtype))
    # 2h1p-1h block
    for iphase in range(nphase):
        for idx in my_indices_map["ipccsd_matvec"][iphase]:
            ki, kj, _ = ktuples[idx]
            kb = kconserv[ki, kshift, kj]
            Hr2[ki, kj] -= einsum('kbij,k->ijb', imds.Wovoo[kshift, kb, ki], r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        logger.info(eom, 'MATVEC Using MP partition for EOM-IP-CCSD')
        fock = imds.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for iphase in range(nphase):
            for idx in my_indices_map["ipccsd_matvec"][iphase]:
                ki, kj, _ = ktuples[idx]
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', fvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', foo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', foo[kj], r2[ki, kj])
    else:
        for iphase in range(nphase):
            Wovov_tmp = fetch_t(my_indices_map, imds.Wovov, kpts, kqrts, rmat, nocc, nvir, "ipccsd_matvec_Wovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
            Wovvo_tmp = fetch_t(my_indices_map, imds.Wovvo, kpts, kqrts, rmat, nocc, nvir, "ipccsd_matvec_Wovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
            for idx in my_indices_map["ipccsd_matvec"][iphase]:
                ki, kj, _ = ktuples[idx]
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', imds.Lvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', imds.Loo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', imds.Loo[kj], r2[ki, kj])
                for kl in range(nkpts):
                    kk = kconserv[ki, kl, kj]
                    Hr2[ki, kj] += einsum('klij,klb->ijb', imds.Woooo[kk, kl, ki], r2[kk, kl])
                    kd = kconserv[kl, kj, kb]
                    Hr2[ki, kj] += 2. * einsum('lbdj,ild->ijb', Wovvo_tmp[kl, kb, kd], r2[ki, kl])
                    Hr2[ki, kj] += -einsum('lbdj,lid->ijb', Wovvo_tmp[kl, kb, kd], r2[kl, ki])
                    Hr2[ki, kj] += -einsum('lbjd,ild->ijb', Wovov_tmp[kl, kb, kj], r2[ki, kl])  # typo in Ref
                    kd = kconserv[kl, ki, kb]
                    Hr2[ki, kj] += -einsum('lbid,ljd->ijb', Wovov_tmp[kl, kb, ki], r2[kl, kj])
            Wovov_tmp = Wovvo_tmp = None
        ##############################################################
        ##############################################################                
        tmp = np.zeros(nvir, dtype=r2.dtype)
        for iphase in range(nphase):    
            for idx in my_indices_map["ipccsd_matvec"][iphase]:
                ki, kj, _ = ktuples[idx]
                tmp += (2. * einsum('klcd,kld->c', imds.Woovv[ki, kj, kshift], r2[ki, kj])
                            - einsum('lkcd,kld->c', imds.Woovv[kj, ki, kshift], r2[ki, kj]))
        comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
        ##############################################################
        ##############################################################
        for iphase in range(nphase):   
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "ipccsd_matvec", t2._dtype, iphase=iphase)
            for idx in my_indices_map["ipccsd_matvec"][iphase]:
                ki, kj, _ = ktuples[idx]
                Hr2[ki, kj] -= einsum('c,ijcb->ijb', tmp, t2_tmp[ki, kj, kshift])
            t2_tmp = None
    safeAllreduceInPlace(comm, Hr2)
    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

def ipccsd_diag(eom, kshift, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    eom.my_indices_map = generate_eom_indices(imds.my_indices_map, imds, kshift)
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv
    my_indices_map = eom.my_indices_map
    ktuples = []
    for ki in range(nkpts): 
        for kj in range(nkpts):
            ktuples.append((ki, kj, kshift))
    kpts = imds.kpts
    kqrts = imds.kqrts
    rmat = imds.rmat
    Hr1 = -np.diag(imds.Loo[kshift])
    Hr1 = np.ascontiguousarray(Hr1)
    comm.Bcast(Hr1, root=0)
    ##############################################################
    ##############################################################
    Hr2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir), dtype=t1.dtype)
    if eom.partition == 'mp':
        logger.info(eom, 'DIAG Using MP partition for EOM-IP-CCSD')
        foo = imds.eris.fock[:, :nocc, :nocc]
        fvv = imds.eris.fock[:, nocc:, nocc:]
        for iphase in range(nphase):
            for i in my_indices_map["ipccsd_diag"][iphase]:
                ki, kj, _ = ktuples[i]
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] = fvv[kb].diagonal()
                Hr2[ki, kj] -= foo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= foo[kj].diagonal()[:, None]
    else:
        idx = np.arange(nocc)
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "ipccsd_diag", t2._dtype, iphase=iphase)
            Wovov_tmp = fetch_t(my_indices_map, imds.Wovov, kpts, kqrts, rmat, nocc, nvir, "ipccsd_diag_Wovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
            Wovvo_tmp = fetch_t(my_indices_map, imds.Wovvo, kpts, kqrts, rmat, nocc, nvir, "ipccsd_diag_Wovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
            for i in my_indices_map["ipccsd_diag"][iphase]:
                ki, kj, _ = ktuples[i]
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] = imds.Lvv[kb].diagonal()
                Hr2[ki, kj] -= imds.Loo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= imds.Loo[kj].diagonal()[:, None]
                if ki == kconserv[ki, kj, kj]:
                    Hr2[ki, kj] += einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:, :, None]
                Hr2[ki, kj] -= einsum('jbjb->jb', Wovov_tmp[kj, kb, kj])
                Wovvo = einsum('jbbj->jb', Wovvo_tmp[kj, kb, kb])
                Hr2[ki, kj] += 2. * Wovvo
                if ki == kj:  # and i == j
                    Hr2[ki, ki, idx, idx] -= Wovvo
                Hr2[ki, kj] -= einsum('ibib->ib', Wovov_tmp[ki, kb, ki])[:, None, :]
                kd = kconserv[kj, kshift, ki]
                Hr2[ki, kj] -= 2. * einsum('ijcb,jibc->ijb', t2_tmp[ki, kj, kshift], imds.Woovv[kj, ki, kd])
                Hr2[ki, kj] += einsum('ijcb,ijbc->ijb', t2_tmp[ki, kj, kshift], imds.Woovv[ki, kj, kd])
            t2_tmp = Wovov_tmp = Wovvo_tmp = None       
    safeAllreduceInPlace(comm, Hr2)
    return eom.amplitudes_to_vector(Hr1, Hr2)
class EOMIP(eom_kgccsd.EOMIP):    
    matvec = ipccsd_matvec
    l_matvec = None
    get_diag = ipccsd_diag
    mask_frozen = eom_kgccsd.mask_frozen_ip
    @property
    def nkpts(self):
        return len(self.kpts.kpts)
    @property
    def ip_vector_desc(self):
        """Description of the IP vector."""
        return [(self.nocc,), (self.nkpts, self.nkpts, self.nocc, self.nocc, self.nmo - self.nocc)]
    def ip_amplitudes_to_vector(self, t1, t2):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]
    def ip_vector_to_amplitudes(self, vec):
        """Ground state vector to amplitudes."""
        return vector_to_nested(vec, self.ip_vector_desc)
    def vector_to_amplitudes(self, vector, kshift=None):
        return self.ip_vector_to_amplitudes(vector)
    def amplitudes_to_vector(self, r1, r2, kshift=None, kconserv=None):
        return self.ip_amplitudes_to_vector(r1, r2)
    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir
    def make_imds(self, eris=None):
        print("Please generating imds before calling ipccsd")
        raise NotImplementedError
########################################
# EOM-EA-CCSD
########################################
def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    nmo = eom.nmo
    t2 = imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv
    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)
    kpts = imds.kpts
    kqrts = imds.kqrts
    rmat = imds.rmat
    comm.Bcast(r1, root=0)
    comm.Bcast(r2, root=0)
    my_indices_map = eom.my_indices_map
    ktuples_std = []
    for ki in range(nkpts): 
        for kj in range(nkpts):
            ktuples_std.append((ki, kj, kshift))
    ##############################################################
    ##############################################################
    ttt = time.perf_counter()
    Hr1 = np.zeros(r1.shape, dtype=np.result_type(imds.Lvv.dtype, r1.dtype))
    base = nkpts // size
    rem = nkpts % size
    xstart = rank * base + min(rank, rem)
    xend = xstart + base + (1 if rank < rem else 0)
    # 1p-2p1h block
    for kl in range(xstart, xend):
        Hr1 += 2. * einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
        Hr1 += -einsum('ld,lda->a', imds.Fov[kl], r2[kl, kl])
        for kc in range(nkpts):
            kd = kconserv[kshift, kc, kl]
            Hr1 += 2. * einsum('alcd,lcd->a', imds.Wvovv[kshift, kl, kc], r2[kl, kc])
            Hr1 += -einsum('aldc,lcd->a', imds.Wvovv[kshift, kl, kd], r2[kl, kc])
    comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
    if rank == 0:
        Hr1 += einsum('ac,c->a', imds.Lvv[kshift], r1)
    comm.Bcast(Hr1, root=0)
    # Eq. (31)
    # 2p1h-1p block
    if gamma_point(eom.kpts.kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    ##############################################################
    ##############################################################
    Hr2 = np.zeros(r2.shape, dtype=np.result_type(dtype, r1.dtype))
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eom_Wvvvo", t2._dtype, iphase=iphase)
        W1ovov_tmp = fetch_t(my_indices_map, imds.W1ovov, kpts, kqrts, rmat, nocc, nvir, "eom_Wvvvo_W1ovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
        W1ovvo_tmp = fetch_t(my_indices_map, imds.W1ovvo, kpts, kqrts, rmat, nocc, nvir, "eom_Wvvvo_W1ovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
        for idx in my_indices_map["r2vvvo"][iphase]:
            kj, ka, _ = ktuples_std[idx]
            kb = kconserv[kshift,ka,kj]
            Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo[ka,kb,kshift,W1ovov_tmp,W1ovvo_tmp,t2_tmp],r1)
        t2_tmp = W1ovov_tmp = W1ovvo_tmp = None
    ##############################################################
    ##############################################################
    # 2p1h-2p1h block
    for iphase in range(nphase):
        for idx in my_indices_map["eaccsd_matvec"][iphase]:
            kj, ka, _ = ktuples_std[idx]
            kb = kconserv[kshift, ka, kj]
            Hr2[kj, ka] -= einsum('lj,lab->jab', imds.Loo[kj], r2[kj, ka])
            Hr2[kj, ka] += einsum('ac,jcb->jab', imds.Lvv[ka], r2[kj, ka])
            Hr2[kj, ka] += einsum('bd,jad->jab', imds.Lvv[kb], r2[kj, ka])
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "r2vvvv", t2._dtype, iphase=iphase)
        for i in my_indices_map["r2vvvv"][iphase]:
            ka, kb, kc, kd = kqrts.kqrts_ibz[i]
            Wvvvv = imds.get_Wvvvv(ka, kb, kc, t2_tmp)
            A, B, C, D = Wvvvv.shape
            AB, CD = A * B, C * D
            Wt = np.ascontiguousarray(Wvvvv.reshape(AB, CD).T)
            op_group = kqrts.stars_ops[i]
            ka_eq_list = kpts.k2opk[ka, op_group]
            kb_eq_list = kpts.k2opk[kb, op_group]
            kc_eq_list = kpts.k2opk[kc, op_group]
            kd_eq_list = kpts.k2opk[kd, op_group]
            for ka_eq, kb_eq, kc_eq, kd_eq, iop in zip(ka_eq_list, kb_eq_list, kc_eq_list, kd_eq_list, op_group):
                Ra = np.ascontiguousarray(rmat.vv[ka][iop].conj())
                Rb = np.ascontiguousarray(rmat.vv[kb][iop].conj())
                Rc = np.ascontiguousarray(rmat.vv[kc][iop])
                Rd = np.ascontiguousarray(rmat.vv[kd][iop])
                kj = kconserv[ka_eq,kshift,kb_eq]
                R2 = r2[kj, kc_eq]
                dj, Cdim, Ddim = R2.shape
                tmp1 = Rc @ np.ascontiguousarray(R2.transpose(1, 0, 2).reshape(Cdim, dj * Ddim))
                tmp1 = tmp1.reshape(Rc.shape[0], dj, Ddim).transpose(1, 0, 2)
                tmp2 = np.ascontiguousarray(tmp1.reshape(dj * Rc.shape[0], Ddim)) @ Rd.T
                X = tmp2.reshape(dj, Rc.shape[0], Rd.shape[0])
                tmp = np.ascontiguousarray(X.reshape(dj, CD)) @ Wt
                Y = tmp.reshape(dj, A, B)
                tmp = (Ra.T @ np.ascontiguousarray(Y.transpose(1, 0, 2).reshape(A, dj * B)))
                tmp = tmp.reshape(Ra.shape[1], dj, B).transpose(1, 0, 2)
                out = np.ascontiguousarray(tmp.reshape(dj * Ra.shape[1], B)) @ Rb
                out = out.reshape(dj, Ra.shape[1], Rb.shape[1])
                Hr2[kj, ka_eq] += out
                # Hr2[kj, ka_eq] += einsum('ABCD,Aa,Bb,Cc,Dd,jcd->jab', Wvvvv, Ra, Rb, Rc, Rd, r2[kj, kc_eq])
        t2_tmp = None
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        Wovov_tmp = fetch_t(my_indices_map, imds.Wovov, kpts, kqrts, rmat, nocc, nvir, "eaccsd_matvec_Wovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
        Wovvo_tmp = fetch_t(my_indices_map, imds.Wovvo, kpts, kqrts, rmat, nocc, nvir, "eaccsd_matvec_Wovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
        for idx in my_indices_map["eaccsd_matvec"][iphase]:
            kj, ka, _ = ktuples_std[idx]
            kb = kconserv[kshift, ka, kj]        
            for kd in range(nkpts):
                kl = kconserv[kd, kb, kj]
                Hr2[kj, ka] += 2. * einsum('lbdj,lad->jab', Wovvo_tmp[kl, kb, kd], r2[kl, ka])
                Hr2[kj, ka] += -einsum('bldj,lad->jab', Wovov_tmp[kl, kb, kj].transpose(1, 0, 3, 2), r2[kl, ka])
                Hr2[kj, ka] += -einsum('bljd,lda->jab', Wovvo_tmp[kl, kb, kd].transpose(1, 0, 3, 2), r2[kl, kd])
                kl = kconserv[kd, ka, kj]
                Hr2[kj, ka] += -einsum('aldj,ldb->jab', Wovov_tmp[kl, ka, kj].transpose(1, 0, 3, 2), r2[kl, kd])
        Wovov_tmp = Wovvo_tmp = None
    ##############################################################
    ##############################################################
    tmp = np.zeros(nocc, dtype=r2.dtype)
    for iphase in range(nphase):            
        for idx in my_indices_map["eaccsd_matvec"][iphase]:
            kj, ka, _ = ktuples_std[idx]
            tmp += (2. * einsum('klcd,lcd->k', imds.Woovv[kshift, kj, ka], r2[kj, ka])
                        - einsum('lkcd,lcd->k', imds.Woovv[kj, kshift, ka], r2[kj, ka]))
    comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
    ##############################################################
    ##############################################################
    for iphase in range(nphase):
        t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eaccsd_matvec", t2._dtype, iphase=iphase)
        for idx in my_indices_map["eaccsd_matvec"][iphase]:
            kj, ka, _ = ktuples_std[idx]
            Hr2[kj, ka] -= einsum('k,kjab->jab', tmp, t2_tmp[kshift, kj, ka]) 
        t2_tmp = None
    safeAllreduceInPlace(comm, Hr2)
    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2, kshift), kshift, const=0.0)

def eaccsd_diag(eom, kshift, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    eom.my_indices_map = generate_eom_indices(imds.my_indices_map, imds, kshift)
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv
    my_indices_map = eom.my_indices_map
    ktuples = []
    for ki in range(nkpts): 
        for kj in range(nkpts):
            ktuples.append((ki, kj, kshift))
    kpts = imds.kpts
    kqrts = imds.kqrts
    rmat = imds.rmat
    Hr1 = np.diag(imds.Lvv[kshift])
    Hr1 = np.ascontiguousarray(Hr1)
    comm.Bcast(Hr1, root=0)
    Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=t2.dtype)
    if eom.partition == 'mp':
        logger.info(eom, 'DIAG Using MP partition for EOM-EA-CCSD')
        foo = imds.eris.fock[:, :nocc, :nocc]
        fvv = imds.eris.fock[:, nocc:, nocc:]
        for iphase in range(nphase):
            for idx in my_indices_map["eaccsd_diag"][iphase]:
                kj, ka, _ = ktuples[idx]
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= foo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += fvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += fvv[kb].diagonal()
    else:
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eaccsd_diag", t2._dtype, iphase=iphase)
            Wovov_tmp = fetch_t(my_indices_map, imds.Wovov, kpts, kqrts, rmat, nocc, nvir, "eaccsd_diag_Wovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
            Wovvo_tmp = fetch_t(my_indices_map, imds.Wovvo, kpts, kqrts, rmat, nocc, nvir, "eaccsd_diag_Wovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
            for idx in my_indices_map["eaccsd_diag"][iphase]:
                kj, ka, _ = ktuples[idx]
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= imds.Loo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += imds.Lvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += imds.Lvv[kb].diagonal()
                Wvvvv = imds.get_Wvvvv(ka, kb, ka, t2_tmp)
                Hr2[kj, ka] += einsum('abab->ab', Wvvvv)
                Hr2[kj, ka] -= einsum('jbjb->jb', Wovov_tmp[kj, kb, kj])[:, None, :]
                Wovvo = einsum('jbbj->jb', Wovvo_tmp[kj, kb, kb])
                Hr2[kj, ka] += 2. * Wovvo[:, None, :]
                if ka == kb:
                    for a in range(nvir):
                        Hr2[kj, ka, :, a, a] -= Wovvo[:, a]
                Hr2[kj, ka] -= einsum('jaja->ja', Wovov_tmp[kj, ka, kj])[:, :, None]
                Hr2[kj, ka] -= 2 * einsum('ijab,ijab->jab', t2_tmp[kshift, kj, ka], imds.Woovv[kshift, kj, ka])
                Hr2[kj, ka] += einsum('ijab,ijba->jab', t2_tmp[kshift, kj, ka], imds.Woovv[kshift, kj, kb])
            t2_tmp = Wovov_tmp = Wovvo_tmp = None        
    safeAllreduceInPlace(comm, Hr2)
    return eom.amplitudes_to_vector(Hr1, Hr2, kshift)
    
class EOMEA(eom_kgccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = None
    get_diag = eaccsd_diag
    mask_frozen = eom_kgccsd.mask_frozen_ea
    @property
    def nkpts(self):
        return len(self.kpts.kpts)
    @property
    def ea_vector_desc(self):
        """Description of the EA vector."""
        nvir = self.nmo - self.nocc
        return [(nvir,), (self.nkpts, self.nkpts, self.nocc, nvir, nvir)]
    def ea_amplitudes_to_vector(self, t1, t2, kshift=None, kconserv=None):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]
    def ea_vector_to_amplitudes(self, vec):
        """Ground state vector to apmplitudes."""
        return vector_to_nested(vec, self.ea_vector_desc)
    def vector_to_amplitudes(self, vector, kshift=None):
        return self.ea_vector_to_amplitudes(vector)
    def amplitudes_to_vector(self, r1, r2, kshift=None, kconserv=None):
        return self.ea_amplitudes_to_vector(r1, r2)
    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nvir + nkpts**2*nocc*nvir*nvir
    def make_imds(self, eris=None):
        print("Please generating imds before calling eaccsd")
        raise NotImplementedError 
    
def Loo(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ki, kl, ka, kb = kq
        if ki == ka:
            fock = (2*einsum('klic,lc->ki', eris.ooov[ki,kl,ki], t1[kl])
                     -einsum('lkic,lc->ki', eris.ooov[kl,ki,ki], t1[kl]))
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[ki][iop]
                Lki[ki] += einsum('ki,km,in->mn', fock, rmat_oo.conj(), rmat_oo)
    return Lki

def Lvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]

    Lac = cc_Fvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned)

    for i in t2_owned:
        kq = kqrts.kqrts_ibz[i]
        ka, kk, kc, kl = kq
        if ka == kc:
            Svovv = 2 * eris.vovv[ka,kk,ka] - eris.vovv[ka,kk,kk].transpose(0,1,3,2)
            fock = einsum('akcd,kd->ac', Svovv, t1[kk])
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_vv = rmat.vv[ka][iop]
                Lac[ka] += einsum('ac,ae,cf->ef', fock, rmat_vv.conj(), rmat_vv)
    return Lac

    
def Woooo(Wklij, kqrts, t1, t2, eris, my_indices):
    nkpts, nocc, nvir = t1.shape
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, kl, ki, kj = kq
        oooo  = einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
        oooo += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
        oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
        oooo += eris.oooo[kk,kl,ki]
        for kc in range(nkpts):
            oooo += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
        Wklij[kk,kl,ki] = oooo

def Wooov(Wklid, kqrts, t1, eris, my_indices):
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, kl, ki, kd = kq
        ooov = einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
        ooov += eris.ooov[kk,kl,ki]
        Wklid[kk,kl,ki] = ooov

def Wovoo(Wkbij, kqrts, t1, t2, eris, kconserv,
          W1ovov, Woooo, W1ovvo, Fov, my_indices):
    nkpts, nocc, nvir = t1.shape
    WW1ovov = W1ovov
    WWoooo = Woooo
    WW1ovvo = W1ovvo
    FFov = Fov
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, kb, ki, kj = kq
        ovoo  = einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
        ovoo += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
        ovoo += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])
        ovoo += np.array(eris.ooov[ki,kj,kk]).transpose(2,3,0,1).conj()
        for kd in range(nkpts):
            kl = kconserv[ki,kk,kd]
            St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
            ovoo += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
            ovoo += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
            kl = kconserv[kb,ki,kd]
            ovoo += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj],t2[kl,ki,kb])
            ovoo += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
        ovoo += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kj],t1[kj],t1[ki])
        ovoo += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
        Wkbij[kk,kb,ki] = ovoo
    

def W1ovov(Wkbid, kqrts, t1, t2, eris, kconserv, my_indices):
    nkpts, nocc, nvir = t1.shape
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, kb, ki, kd = kq
        ovov = eris.ovov[kk,kb,ki].copy()
        for kl in range(nkpts):
            kc = kconserv[kk,kd,kl]
            ovov -= einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
        Wkbid[kk,kb,ki] = ovov

def W1ovvo(Wkaci, kqrts, t1, t2, eris, kconserv, my_indices):
    nkpts, nocc, nvir = t1.shape
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, ka, kc, ki = kq
        ovvo = np.asarray(eris.voov[ka,kk,ki]).transpose(1,0,3,2).copy()
        for kl in range(nkpts):
            kd = kconserv[ki,ka,kl]
            St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
            ovvo +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
            ovvo += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
        Wkaci[kk,ka,kc] = ovvo

def Wovov(Wooov, Wovov_out, kqrts, t1, eris, my_indices):
    # my_indices should be W1ovov's indices
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, kb, ki, kd = kq
        ovov = einsum('klid,lb->kbid',Wooov[kk,kb,ki],-t1[kb])
        ovov += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
        Wovov_out[kk,kb,ki] += ovov
        
def Wovvo(Wooov, Wovvo_out, kqrts, t1, eris, my_indices):
    # my_indices should be W1ovvo's indices
    for i in my_indices:
        kq = kqrts.kqrts_ibz[i]
        kk, ka, kc, ki = kq
        ovvo = einsum('la,lkic->kaci',-t1[ka],Wooov[ka,kk,ki])
        ovvo += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
        Wovvo_out[kk,ka,kc] += ovvo
        
        
class _IMDS:
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self.kconserv = cc.khelper.kconserv
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False
        self.kpts = cc.kpts
        self.kqrts = cc.kqrts
        self.rmat = cc.rmat
        self.t2_owned = cc.t2_owned
        self.t2_tuples = cc.t2_tuples
        self.my_indices_map = {}
        self.my_indices_map = generate_eom_indices(self.my_indices_map, self)
        
    
    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        kpts, kqrts, rmat = self.kpts, self.kqrts, self.rmat
        nkpts, nocc, nvir = t1.shape
        t2_owned = self.t2_owned
        self.Loo = Loo(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
        comm.Allreduce(MPI.IN_PLACE, self.Loo.data, op=MPI.SUM)
        for i in range(kpts.nkpts_ibz):
            ki = kpts.ibz2bz[i]
            self.Loo[ki] += eris.fock[ki,:nocc,:nocc]
            self.Loo[ki] += einsum('kc,ic->ki', eris.fock[:,:nocc,nocc:][ki], t1[ki])
        self.Loo.todense()
        self.Lvv = Lvv(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
        comm.Allreduce(MPI.IN_PLACE, self.Lvv.data, op=MPI.SUM)
        for ka_ibz in range(kpts.nkpts_ibz):
            ka = kpts.ibz2bz[ka_ibz]
            self.Lvv[ka] += eris.fock[ka,nocc:,nocc:]
            self.Lvv[ka] += -einsum('kc,ka->ac', eris.fock[:,:nocc,nocc:][ka], t1[ka])
        self.Lvv.todense()
        self.Fov = cc_Fov(kpts, kqrts, t1, t2, eris, rmat, t2_owned)
        comm.Allreduce(MPI.IN_PLACE, self.Fov.data, op=MPI.SUM)
        for i in range(kpts.nkpts_ibz):
            ki = kpts.ibz2bz[i]
            self.Fov[ki] += eris.fock[ki,:nocc,nocc:]
        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        kpts, kqrts, rmat = self.kpts, self.kqrts, self.rmat
        my_indices_map = self.my_indices_map
        nkpts, nocc, nvir = t1.shape
        # 2 virtuals
        ##############################################################
        ##############################################################
        W1ovov_total_indices = [x for i in my_indices_map["eom_W1ovov"] for x in i]
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'ovov', 'trans': 'ccnn',
                    'incore': True, 'owned_m': W1ovov_total_indices}
        self.W1ovov = zeros([nocc,nvir,nocc,nvir], dtype=t1.dtype, metadata=metadata)
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eom_W1ovov", t2._dtype, iphase=iphase)
            W1ovov(self.W1ovov, kqrts, t1, t2_tmp, eris, kconserv, my_indices_map["eom_W1ovov"][iphase])
            t2_tmp = None
        ##############################################################
        ##############################################################
        W1ovvo_total_indices = [x for i in my_indices_map["eom_W1ovvo"] for x in i]
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 
                    'label': 'ovvo', 'trans': 'ccnn', 
                    'incore': True, 'owned_m': W1ovvo_total_indices}
        self.W1ovvo = zeros([nocc,nvir,nvir,nocc], dtype=t1.dtype, metadata=metadata)
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eom_W1ovvo", t2._dtype, iphase=iphase)
            W1ovvo(self.W1ovvo, kqrts, t1, t2_tmp, eris, kconserv, my_indices_map["eom_W1ovvo"][iphase])
            t2_tmp = None
        log.timer('EOM-CCSD W1ovov, W1ovvo intermediates', *cput0)
        ##############################################################
        ##############################################################
        self.Wovov = copy.deepcopy(self.W1ovov)
        Wovov(self.Wooov, self.Wovov, kqrts, t1, eris, W1ovov_total_indices)
        self.Wovvo = copy.deepcopy(self.W1ovvo)
        Wovvo(self.Wooov, self.Wovvo, kqrts, t1, eris, W1ovvo_total_indices)
        self.Woovv = eris.oovv
        
    def make_ip_ea(self, partition=None):
        # Here I assume ip and ea share the same partition scheme.
        # IP
        log = logger.Logger(self.stdout, self.verbose)
        log.info("SL: Making intermediates for IP/EA-EOM-CCSD......")
        
        cput0 = (logger.process_clock(), logger.perf_counter())
        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        kpts, kqrts, rmat = self.kpts, self.kqrts, self.rmat
        my_indices_map = self.my_indices_map
        nkpts, nocc, nvir = t1.shape
        ##############################################################
        ##############################################################
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 
                    'label': 'oooo', 'trans': 'ccnn', 'incore': True}
        self.Woooo = zeros([nocc,nocc,nocc,nocc], dtype=t1.dtype, metadata=metadata)
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eom_Woooo", t2._dtype, iphase=iphase)
            Woooo(self.Woooo, kqrts, t1, t2_tmp, eris, my_indices_map["eom_Woooo"][iphase])
            t2_tmp = None
        comm.Allreduce(MPI.IN_PLACE, self.Woooo.data, op=MPI.SUM)
        ##############################################################
        ##############################################################
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 
                    'label': 'ooov', 'trans': 'ccnn','incore': True}
        self.Wooov = zeros([nocc,nocc,nocc,nvir], dtype=t1.dtype, metadata=metadata)
        Wooov(self.Wooov, kqrts, t1, eris, my_indices_map["eom_Wooov"])
        safeAllreduceInPlace_ksymm(comm, self.Wooov)
        log.timer('EOM-CCSD Woooo, Wooov intermediates', *cput0)
        ##############################################################
        ##############################################################
        self._make_shared_1e()
        if self._made_shared_2e is False and partition != 'mp':
            self._make_shared_2e()
            self._make_shared_2e = True
        ##############################################################
        ##############################################################
        cput1 = (logger.process_clock(), logger.perf_counter())
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat, 
                    'label': 'ovoo', 'trans': 'ccnn', 'incore': True}
        self.Wovoo = zeros([nocc,nvir,nocc,nocc], dtype=t1.dtype, metadata=metadata)
        for iphase in range(nphase):
            t2_tmp = fetch_t(my_indices_map, t2, kpts, kqrts, rmat, nocc, nvir, "eom_Wovoo", t2._dtype, iphase=iphase)
            W1ovov_tmp = fetch_t(my_indices_map, self.W1ovov, kpts, kqrts, rmat, nocc, nvir, "eom_Wovoo_W1ovov", t2._dtype, iphase=iphase, label='ovov', trans='ccnn')
            W1ovvo_tmp = fetch_t(my_indices_map, self.W1ovvo, kpts, kqrts, rmat, nocc, nvir, "eom_Wovoo_W1ovvo", t2._dtype, iphase=iphase, label='ovvo', trans='ccnn')
            Wovoo(self.Wovoo, kqrts, t1, t2_tmp, eris, kconserv, W1ovov_tmp, 
                  self.Woooo, W1ovvo_tmp, self.Fov, my_indices_map["eom_Wovoo"][iphase])
            t2_tmp = W1ovov_tmp = W1ovvo_tmp = None
        safeAllreduceInPlace_ksymm(comm, self.Wovoo)
        log.timer('EOM-CCSD Wovoo intermediates', *cput1)
        ##############################################################
        ##############################################################
        # EA
        cput2 = (logger.process_clock(), logger.perf_counter())
        def _get_Wvovv(ka, kl, kc):
            vovv = einsum('ka,klcd->alcd', -t1[ka], eris.oovv[ka,kl,kc])
            vovv += eris.vovv[ka,kl,kc]
            return vovv
        self.Wvovv = compute_block(funct=lambda ka, kl, kc: _get_Wvovv(ka, kl, kc))
        ##############################################################
        ##############################################################
        def _get_Wvvvo(ka,kb,kc,WW1ovov,WW1ovvo,t2tmp):
            nkpts, nocc, nvir = t1.shape
            FFov = self.Fov
            kj = kconserv[ka,kc,kb]
            vvvo  = einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
            vvvo += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])
            vvvo += np.asarray(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
            for kl in range(nkpts):
                kd = kconserv[ka,kc,kl]
                St2 = 2.*t2tmp[kl,kj,kd] - t2tmp[kl,kj,kb].transpose(0,1,3,2)
                vvvo += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                vvvo += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2tmp[kl,kj,kd])
                kd = kconserv[kb,kc,kl]
                vvvo += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2tmp[kj,kl,kd])
                kk = kconserv[kb,kl,ka]
                vvvo += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2tmp[kl,kk,kb])
            vvvo += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
            vvvo += einsum('lc,ljab->abcj',-FFov[kc],t2tmp[kc,kj,ka]) 
            # Check if t1=0 (HF+MBPT(2))
            if np.any(t1.data != 0):
                kj = kconserv[ka,kc,kb]
                Wvvvv = self.get_Wvvvv(ka, kb, kc, t2tmp)
                vvvo += einsum('abcd,jd->abcj', Wvvvv, t1[kj]) 
            return vvvo
        self.Wvvvo = compute_block(funct=lambda ka,kb,kc,WW1ovov,WW1ovvo,t2tmp: _get_Wvvvo(ka,kb,kc,WW1ovov,WW1ovvo,t2tmp))
        log.timer('EOM-CCSD EA intermediates (on-the-fly)', *cput2)

    def get_Wvvvv(self, ka, kb, kc, t2tmp):
        t1, eris = self.t1, self.eris
        kconserv = self.kconserv
        kd = kconserv[ka, kc, kb]
        nkpts, nocc, nvir = t1.shape
        Lpq = eris.Lpq
        Lac = (Lpq[ka,kc][:,nocc:,nocc:] -
                einsum('Lkc,ka->Lac', Lpq[ka,kc][:,:nocc,nocc:], t1[ka]))
        Lbd = (Lpq[kb,kd][:,nocc:,nocc:] -
                einsum('Lkd,kb->Lbd', Lpq[kb,kd][:,:nocc,nocc:], t1[kb]))
        vvvv = einsum('Lac,Lbd->abcd', Lac, Lbd)
        vvvv *= (1. / nkpts)
        for kk in range(nkpts):
            kl = kconserv[kc,kk,kd]
            vvvv += einsum('klcd,klab->abcd', eris.oovv[kk,kl,kc], t2tmp[kk,kl,ka])
        return vvvv


#############################################################
# Example driver
#############################################################


def run_lih_example(
    ikpt=2,
    basis="basis/cc-pvdz.dat",
    auxbasis="basis/cc-pvdz-jkfit.dat",
    save_dir=None,
    max_memory=1400000,
    cc_max_memory=1400000,
    max_cycle=200,
):
    from pyscf.pbc import gto, scf
    a = 4.083
    shift = a / 2
    cell = gto.Cell()
    cell.unit = "angstrom"
    cell.a = np.array([
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ])
    cell.atom = f"""
    Li  0.00000000  0.00000000  0.00000000
    H   {shift:.8f}  {shift:.8f}  {shift:.8f}
    """
    cell.basis = basis
    cell.pseudo = "gth-hf-rev"
    cell.spin = 0
    cell.verbose = 4 if rank == 0 else 0
    cell.exp_to_discard = 0.0
    cell.space_group_symmetry = True
    cell.max_memory = max_memory
    cell.precision = 1e-14
    cell.build()

    kmesh_shift = [0.5, 0.0, 0.5]
    nk = [ikpt, ikpt, ikpt]
    kpts = cell.make_kpts(nk, scaled_center=kmesh_shift, space_group_symmetry=True)

    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit(auxbasis=auxbasis)
    kmf.verbose = 4 if rank == 0 else 0
    kmf.with_df.build()

    if rank == 0:
        kmf.kernel()
        report_mem("SCF")
        # np.savez(
        #     "idx_tag_hf.npz",
        #     mo_coeff=kmf.mo_coeff,
        #     mo_energy=kmf.mo_energy,
        #     mo_occ=kmf.mo_occ,
        #     e_tot=kmf.e_tot,
        # )

    comm.Barrier()
    kcc = RCCSD(kmf)
    kcc.max_memory = cc_max_memory
    kcc.verbose = 7 if rank == 0 else 0
    kcc.max_cycle = max_cycle
    kcc.save_dir = save_dir
    kcc.save_per_iter = False
    ecc, t1, t2 = kcc.kernel()

    if rank == 0:
        print("cc energy (per unit cell) = %.17g" % ecc)
    ekip, wkip = kcc.ipccsd(nroots=1, koopmans=True, kptlist=[0])
    ekea, wkea = kcc.eaccsd(nroots=1, koopmans=True, kptlist=[0])
    if rank == 0:
        print("IP energy = %.17g" % ekip[0][0])
        print("EA energy = %.17g" % ekea[0][0])
    
        assert np.isclose(ecc, -0.057645651053093563, atol=1e-8)
        assert np.isclose(ekip[0][0], -0.1376236923924606, atol=1e-8)
        assert np.isclose(ekea[0][0], 0.2777348901603924, atol=1e-8)

if __name__ == "__main__":
    run_lih_example()
