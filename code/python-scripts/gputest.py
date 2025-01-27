"""Module gputest.py tests speed of some dense and sparse matrix operations on
CPU and/or GPU.

Usage:
    export gpu=1            # (optional) use GPU 1 (2nd GPU)
    export gpu=-1           # (optional) don't import cupy, don't use GPU
    python gputest.py       # run tests
If environment variable gpu is not set, uses default GPU.  If unable to import
cupy or to initialize the specified GPU, runs tests on CPU only. Constant CPU
can be set to False to run tests on GPU only.

Detailed description of tests:

Except for the first test 'dense_mult', these tests model operations required
for a DEM-type simulaton of N nodes in one spatial dimension with B=Q*N pairs
of nodes connected by bonds. Runs tests for several N values as set by the
constant NS, while Q=B/N is a constant. 

Constant TESTS should list one or more of the tests listed below, each of
which is carried out for the floating point types listed in FPTYPES and the N
values listed in NS.  For each (test,FPTYPE,N) the input and zeroed output
arrays are constructed only once and timed separately, then the specified test
operation on the arrays are carried out multiple times on the CPU and/or GPU
and the average timings are reported.

For GPU tests the input+output arrays are transferred to the GPU only once
(before running the test calculation several times) and the output array is
transferred back to the CPU only once.  The difference betwen CPU and GPU
elapsed time is used to infer the CPU-GPU transfer speed.

Here are the available tests. In the following int is always int32, while
float is the FP type (float64, float32..) being tested from FPTYPES:

'dense_mult' tests the multiplication of MxM dense float matrices, with N
rounded to give an integer maxtrix size M with M*M = N.
    
'drb_rn_np' tests the calculation of bb=B bond vectors from nn=N 1D node
positions, with each bond vector the difference between two node positions.
Uses numpy integer array indexing to retrieve the positions.

'drb_rn_csri' tests exactly the same calculation, but uses a CSR sparse matrix
with integer elements 1, -1 to retrieve and subtract the positions (can't
currently do on GPU).

'drb_rn_csrf' tests exactly the same calculation, but uses a CSR sparse matrix
with float elements 1.0, -1.0.

'fn_fb_at' tests the calculation of N node forces from B 1D bond forces, with
each node force the sum (with weights +/-1) of several (number varying by
node) bond forces. Uses the ufunc methods np.add.at, np.subtract.at to
accumulate the forces on each node (can't currently do on GPU)..

'fn_fb_csrf' tests exactly the same calculation, but uses a CSR sparse matrix
with float elements 1.0, -1.0 to retrieve and accumulate the node forces.
"""
THIS_IS = 'gputest.py 10/29a/23 D.C.'

CPU = True              # whether to test CPU speed
# CPU = False
TESTS = 'dense_mult','drb_rn_np',    # tests to run
TESTS = ('dense_mult','drb_rn_np','drb_rn_csri','drb_rn_csrf','fn_fb_at',
         'fn_fb_csrf')  #  all possible tests
FPTYPES = 'float32',    # float types to try
FPTYPES = 'float64','float32'
NS = 1_000,10_000        # N values (number of mat elements or nodes) to test
NS = 100_000,1_000_000
NS = 100_000,1_000_000,10_000_000 

Q = 3                   # B/N (bonds per node), must be int
TSKIP = 1               # For CPU test calculations >=0 to skip at start ...
TUSE = 10               # ... and test calculations to use for timing
TSKIP_GPU = 3
TUSE_GPU = 25           # same quantities for GPU
SEED = 2023             # seed for random matrix/vec elements and bond indices
# SEED = None             # use unseeded randoms

# Dict of GPU models guessed from (compute capacity,SMs).
GPUMODELS = {('6.1',6) : 'GeForce GTX 1050',
             ('7.5',22) : 'GeForce GTX 1660',
             ('7.5',40) : 'Tesla T4',
             ('7.0',80) : 'Tesla V100',
             ('8.0',108) : 'Tesla A100',
             ('8.0',144) : 'Hopper H100'}
# Dict of all possible tests and descriptive strings.
ALLTESTS = {'dense_mult' : 'Multiply M*M=N element dense matrices',
            'drb_rn_np' : (f'Calc {Q}N bond vecs from N 1D node pos'
                          ' using array indexing'),
            'drb_rn_csri' : (f'Calc {Q}N bond vecs from N 1D node pos'
                           ' using int CSR matrix'),
            'drb_rn_csrf' : (f'Calc {Q}N bond vecs from N 1D node pos'
                           ' using float CSR matrix'),
            'fn_fb_at' : (f'Calc N node forces from {Q}N 1D bond forces'
                          ' using ufunc.at'),
            'fn_fb_csrf' : (f'Calc N node forces from {Q}N 1D bond forces'
                           ' using float CSR matrix')}
# Messages about tests unsupported on GPU.
UNSUPGPU = {'drb_rn_csri' : ('Can\'t do on GPU, cupyx.scipy.sparse.csr_matrix'
                             ' does not support int.'),
            'fn_fb_at' : ('Can\'t do on GPU, cupy.subtract.at does not support'
                          ' float.')}
 
FPBYTES = {'float64':8,'float32':4}   # bytes in each float type

import numpy as np
from numpy import sqrt
from scipy import sparse
import os,time
# Use environment variable 'gpu' to decide whether and which GPU to use.
gpuinfo = ''
gpu = True
device = os.environ.get('gpu')
if device is not None: 
    device = int(device)
    if device<0:
        gpuinfo = 'Environment var gpu<0, using CPU only'
        gpu = False
if gpu:
    try:
        import cupy as cp        # numpy-equiv and CUDA GPU funcs        
        import cupyx             # scipy-equiv GPU funcs        
        try:
            gpudevice = cp.cuda.Device(device)
            gpuid = gpudevice.id
            gpuccap = gpudevice.compute_capability
            gpuccstr = f'{gpuccap[0]}.{gpuccap[1]}'
            gpusms = gpudevice.attributes["MultiProcessorCount"]
            gpuram = gpudevice.mem_info[1]
            gpumodel = GPUMODELS.get((gpuccstr,gpusms))
            gpuinfo = (f'GPU {gpuid} has compute capacity {gpuccstr},'
                       f' {gpusms} SMs, {gpuram/1e9:.2f} GB RAM,'
                       f' guess model = {gpumodel}')
        except cp.cuda.runtime.CUDARuntimeError:
            gpuinfo = (f'Could not initialize GPU with device={device},'
                       ' using CPU only')
            gpu = False
    except:
        gpuinfo = 'Import cupy failed, using CPU only'
        gpu = False

def get_mats(test,fptype,nn):
    """Get the input matrices (filled in) and output matrix (zero-filled) for
    a test.
    
    Parameters
    ----------
    test : str
        Test being done (must be in ALLTESTS).
    fptype : str
        'float64','float32'...  = floating-point type to use.
    nn : int
        Requested number of dense matrix elements or nodes N, may be adjusted.
            
    Returns
    -------
    mats : dict with following entries.
        'nn' : int
            nn actually used for tests, possibly adjusted.
        'flop' : int
            Number of floating-point operations in one trial of test.
        'xbytes' : int
            Total bytes transferred, when all arrays in mats are transferred
            from CPU to GPU and mats['output'] is transferred back to CPU.
        'tmats' : float
            Time (sec) required to run this function.
        The remaining entries in mats are dense and/or sparse arrays added
        according to test by mats_dense, mats_drb, or mats_fn (see). The array
        used to hold the test output is always called 'output'.
    """
    t0 = time.perf_counter()
    # Call funcs that add 'nn','flop', 'xbytes', and matrices to mats.
    mats = {}
    mats_dense(test,fptype,nn,mats)   # for test='dense_mult'
    mats_drb(test,fptype,nn,mats)     # for test=drb_rn_...'
    mats_fn(test,fptype,nn,mats)      # for test='fn_fb_...'
    mats['tmats'] = time.perf_counter() - t0
    return mats

def mats_dense(test,fptype,nn,mats):
    """Helper for get_mats: Makes matrices for test='dense_mult'
    (multplication of two square dense matrices).
    
    Parameters
    ----------
    test,fptype,nn
        As passed to get_mats (see).
    mats : dict
    
    Side effects
    ------------
    When test='dense_mult', adds following entries to mats.
    
    'nn' : int
        Adjusted from nn to a perfect square, mats['nn'] = mm*mm with int mm.
    'flop','xbytes' : int
        FP ops in one trial, total bytes transferred to and from GPU.
    'aa','bb' : array (mm,mm) of fptype
        Arrays to be multiplied, filled with gaussian randoms.
    'output' : array (mm,mm) of fptype
        Array for test output, filled with zeros. Test will set to aa@bb.
    """
    if not test=='dense_mult':
        return
    mm = round(sqrt(nn))              # linear size M of matrices
    mats['nn'] = mm*mm
    mats['flop'] = mm*mm*(2*mm-1)     # FP mults and adds in one matrix mult
    rng = np.random.default_rng(SEED)
    mats['aa'] = rng.normal(size=(mm,mm)).astype(fptype)
    mats['bb'] = rng.normal(size=(mm,mm)).astype(fptype)
    mats['output'] = np.zeros((mm,mm),fptype)
    fpbytes = FPBYTES[fptype]          # bytes per float for type in use
    matels = mm*mm                     # elements in a dense mm*mm matrix
    mats['xbytes'] = 3*fpbytes*matels  # xfer 'aa','bb','output' to GPU
    mats['xbytes'] += fpbytes*matels   # xfer 'output' back to CPU

def mats_drb(test,fptype,nn,mats):
    """Helper for get_mats: Makes matrices for test='drb_rn_..' calculation of
    bb=Q*nn bond distances as differences between bb pairs of node positions).
    
    Parameters
    ----------
    test,fptype,nn
        As passed to get_mats (see).
    mats : dict
    
    Side effects
    ------------
    When test='drb_rn_..', adds following entries to mats.
        'nn' : int
            Set to nn (number of nodes N).
        'flop','xbytes' : int
            FP ops in one trial, total bytes transferred to and from GPU.
        'x' : array (nn) of fptype
            1D positions of the nodes, filled with gaussian randoms.
        'output' : array (bb) of float
            Array for test output, filled with zeros. Test will set to
            x[nb2]-x[nb1] with nb1,nb2 as generated by get_nb12 (see).     
    Addtionally for test='drb_rn_np':
        'nb1', 'nb2'
            nb1,nb2 as generated by get_nb12 (see).     
    Addtionally for test='drb_rn_csri':
        'nbmat_csri' : scipy.sparse.csr_array (bb,nn) of int32
            CSR-format sparse array with two elements in each row, a +1 and a
            -1 in the columns giving the nodes for the ends of each bond.
    Addtionally for test='drb_rn_csrf':
        'nbmat_csrf' : scipy.sparse.csr_array (bb,nn) of fptype
            CSR-format sparse array with two elements in each row, a +1.0 and
            a -1.0 in the columns giving the nodes for the ends of each bond.
    """
    if not test.startswith('drb_rn'):
        return
    mats['nn'],bb = nn,Q*nn
    mats['flop'] = bb              # each of bb bonds requires one FP subtract
    rng = np.random.default_rng(SEED)
    nb1,nb2 = get_nb12(nn,rng)     # get random node-to-bond connectivity
    mats['x'] = rng.normal(size=nn).astype(fptype)
    mats['output'] = np.zeros(bb,fptype)
    fpbytes = FPBYTES[fptype]          # bytes per float for type in use
    mats['xbytes'] = fpbytes*(nn+bb)   # xfer 'x','output' to GPU
    mats['xbytes'] += fpbytes*bb       # xfer 'output' back to CPU
    if test=='drb_rn_np':
        mats['nb1'] = nb1
        mats['nb2'] = nb2
        nbbytes = 4*bb                 # nb1,nb2 each have bb int32 elements
        mats['xbytes'] += 2*nbbytes    # xfer 'nb1','nb2' to GPU
    else:
        # Will make a sparse matrix, start with COO matrix.
        nbmat_coo = get_coo(nn,nb1,nb2)
        matels = 2*bb                  # sparse arrays used have 2*bb elements 
        if test=='drb_rn_csri':
            # Make int CSR matrix.
            mats['nbmat_csri'] = nbmat_coo.tocsr()
            # xfer 'nbmat_crsi' to GPU with matels int32's
            mats['xbytes'] += 4*matels
        elif test=='drb_rn_csrf':
            # Make float CSR matrix.
            mats['nbmat_csrf'] = nbmat_coo.tocsr().astype(fptype)
            # xfer 'nbmat_crsf' to GPU with matels floats
            mats['xbytes'] += fpbytes*matels
     
def mats_fn(test,fptype,nn,mats):
    """Helper for get_mats: Makes matrices for test='fn_fb_..' (calculation of
    nn node forces obtained by summing the forces in bb=Q*nn bonds).
    
    Parameters
    ----------
    test,fptype,nn
        As passed to get_mats (see).
    mats : dict
    
    Side effects
    ------------
    When test='fn_fb_..', adds following entries to mats.
        'nn' : int
            Set to nn (number of nodes N).
        'flop','xbytes' : int
            FP ops in one trial, total bytes transferred to and from GPU.
        'fb' : array (bb) of fptype
            1D forces in the bonds, filled with gaussian randoms.
        'output' : array (nn) of float
            Array for test output, filled with zeros. Test will set to summed
            forces of bonds connecting to each node with weights +/-1.
    Addtionally for test='fn_fb_at':
        'nb1', 'nb2'
            nb1,nb2 as generated by get_nb12 (see).     
    Addtionally for test='fn_fb_csrf':
        'nbmat2_csrf' : scipy.sparse.csr_array (nn,bb) of fptype
            CSR-format sparse array with +1.0 and -1.0 elements.  n'th row has
            +/-1.0 in columns for bonds with forces that should be
            added/subtracted to get force on n'th node.  Each row will have on
            average 2Q elements. ('nbmat2_csrf' is the transpose of
            'nbmat_csrf' made by mats_drb.)
    """
    if not test.startswith('fn_fb'):
        return
    mats['nn'],bb = nn,Q*nn
    mats['flop'] = 2*bb       # for each of bb bonds do an FP add and subtract
    rng = np.random.default_rng(SEED)
    nb1,nb2 = get_nb12(nn,rng)        # get random node-to-bond connectivity
    mats['fb'] = rng.normal(size=bb).astype(fptype)
    mats['output'] = np.zeros(nn,fptype)
    fpbytes = FPBYTES[fptype]         # bytes per float for type in use
    mats['xbytes'] = fpbytes*(bb+nn)  # xfer 'fb','output' to GPU
    mats['xbytes'] += fpbytes*nn      # xfer 'output' back to CPU
    if test=='fn_fb_at':
        mats['nb1'] = nb1
        mats['nb2'] = nb2
        nbbytes = 4*bb                 # nb1,nb2 each have bb int32 elements
        mats['xbytes'] += 2*nbbytes    # xfer 'nb1','nb2' to GPU
    elif test=='fn_fb_csrf':
        # Will make a sparse matrix, start with COO matrix.
        nbmat_coo = get_coo(nn,nb1,nb2)
        matels = 2*bb              # sparse arrays used have 2*bb elements 
        # Make float CSR matrix.
        mats['nbmat2_csrf'] = nbmat_coo.transpose().tocsr().astype(fptype)
        # xfer 'nbmat2_crsf' to GPU with matels floats
        mats['xbytes'] += fpbytes*matels
    
def get_nb12(nn,rng):
    """Helper for mats_drb, mats_fb: Generates random node indices for the
    two ends of bb=Q*nn bonds between pairs of nodes.
    
    Parameters
    ----------
    nn : int
        Number N of nodes in the system.
    rng : Generator
        Random number generator in use.
        
    Returns
    -------
    nb1,nb2 : array (bb) of int32
        Node indices for the starts end ends of bb bonds, with bb=Q*nn. Filled
        with randoms in 0..nn-1.
    """
    bb=Q*nn
    nb1 = rng.integers(nn,size=bb,dtype='int32')
    nb2 = rng.integers(nn,size=bb,dtype='int32')
    return nb1,nb2

def get_coo(nn,nb1,nb2):
    """Helper for mats_drb, mats_fb: Makes COO sparse matrix used to compute
    CSR sparse matrix used for trials.
    
    Parameters
    ----------
    nn : int
        Number N of nodes in the system.
    nb1,nb2
        As returned by get_nb12 (see).
        
    Returns
    -------
    nbmat_coo: scipy.sparse.csr_array (bb,nn) of int32
        COO-format sparse array with two elements in each row, a +1 and a
        -1 in the columns giving the nodes for the ends of each bond.
    """
    bb = Q*nn
    bb1s = np.ones(bb,'int32')
    bbm1s = -bb1s
    dat = np.concatenate((bb1s,bbm1s))   # matrix elements for COO array
    bbrng = np.arange(bb,dtype='int32')
    rows = np.concatenate((bbrng,bbrng)) # row indices for COO array
    cols = np.concatenate((nb2,nb1))     # col indices for COO array
    nbmat_coo = sparse.coo_array((dat,(rows,cols)),shape=(bb,nn))
    return nbmat_coo

def cpu_test(test,mats):
    """Carry out and time a test on the CPU.
    
    Parameters
    ----------
    test : str
        Test being done, must be in ALLTESTS.
    mats : dict of arrays
        Input and ouput arrays for this test as returned by get_mats (see).
        
    Side effect
    -----------
    Fills mats['output'] (1D or 2D float array depending on test) with output
    from last trial of test.
        
    Returns
    -------
    t_cpu : float
        Average time to complete one test (s).  Will do TSKIP trials without
        timing, then will do TUSE trials and return average time taken.
    output : array of float
        1D or 2D (depending on test) array giving test output.
    """
    for j in range(TSKIP):
        cpu_test1(test,mats)                  # do TSKIP untimed trials
    t0 = time.perf_counter()
    for j in range(TUSE):
        cpu_test1(test,mats)                  # do TUSE timed trials...
    t_cpu = (time.perf_counter() - t0)/TUSE   # ..and compute av time/trial
    return t_cpu

def cpu_test1(tests,mats):
    """Helper for cpu_trial, carries out a trial for test. Parameters
    test,mats are as specified for cpu_test.
    """
    output = mats['output']
    if test=='dense_mult':
        # Dense-matrix multiplication.
        np.matmul(mats['aa'],mats['bb'],out=output)
    elif test in ('drb_rn_np','drb_rn_csri','drb_rn_csrf'):
        # Calculation of bond vectors from 1D node positions...
        x = mats['x']
        if test=='drb_rn_np':
            # ...using array indexing, or...
            output[:] = x[mats['nb2']] - x[mats['nb1']]
        elif test=='drb_rn_csri':
            # ...using int CSR matrix, or..
            output[:] = mats['nbmat_csri'].dot(x)
        elif test=='drb_rn_csrf':
            # ...using float CSR matrix.
            output[:] = mats['nbmat_csrf'].dot(x)
    elif test in ('fn_fb_at','fn_fb_csrf'):
        # Calculation of node forces from 1D bond forces...
        fb = mats['fb']
        if test=='fn_fb_at':
            # ...using ufunc.at, or...
            output.fill(0.)
            np.add.at(output,mats['nb2'],fb)
            np.subtract.at(output,mats['nb1'],fb)
        elif test=='fn_fb_csrf':
            # ...using float CSR matrix.
            output[:] = mats['nbmat2_csrf'].dot(fb)
            
def gpu_test(test,mats):
    """Carry out and time a test on the GPU. (Do not call for tests in
    UNSUPGPU as these result in cupy calls with unsupported data types.)
    
    Parameters
    ----------
    test : str
        Test being done, must be in ALLTESTS.
    mats : dict of arrays
        Input and ouput arrays for this test as returned by get_mats (see).
        
    Side effect
    -----------
    Fills mats['output'] (1D or 2D float array depending on test) with output
    from last trial of test.

    Returns
    -------
    t_gpu : float
        Average time to complete one test (s).  Will do TSKIP_GPU trials
        without timing, then will do TUSE_GPU trials and return average time
        taken.
    tx: float
        Total time (s) used to transfer matrices between CPU and GPU.
    """
    # Get CUDA events, will use to mark start/end of calcs on GPU.
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev2 = cp.cuda.Event()
    mats['output'].fill(0.0)    # ensures won't see leftover CPU output
    tcpu0 = time.perf_counter()
    # Transfer input and output arrays from CPU to GPU.
    mats_gpu = {}
    for key,val in mats.items():
        # Transfer each numpy array in mats to a cupy array on GPU.
        if isinstance(val,np.ndarray):
            mats_gpu[key] = cp.asarray(val)
        # Transfer each scipy CSR array in mats to a cupyx CSR array on GPU.
        if isinstance(val,sparse._csr.csr_array):
            mats_gpu[key] = cupyx.scipy.sparse.csr_matrix(val)
    ev0.record()                      # mark start of untimed trials
    for j in range(TSKIP_GPU):
        gpu_test1(test,mats_gpu)      # do TSKIP_GPU untimed trials
    ev1.record()                      # mark start of timed trials
    for j in range(TUSE_GPU):
        gpu_test1(test,mats_gpu)      # do TUSE_GPU timed trials
    ev2.record()                      # mark end of timed trials
    # Transfer output array back to CPU
    # mats['output'] = mats_gpu['output'].get()
    mats_gpu['output'].get(out=mats['output'])  # better? only with pinned?
    # ev2.synchronize()                 # not needed, I think
    tcpu = time.perf_counter() - tcpu0   # total CPU time
    # Compute GPU time per test as (time between ev1 and ev2)/(number of tests)
    tgpu12 = 1e-3*cp.cuda.get_elapsed_time(ev1,ev2)
    t_gpu = tgpu12/TUSE_GPU
    # Compute transfer time as (total CPU time) - (time between ev0 and ev2)
    tgpu02 = 1e-3*cp.cuda.get_elapsed_time(ev0,ev2)
    tx = tcpu - tgpu02
    return t_gpu,tx

def gpu_test1(test,mats_gpu):
    """Helper for gpu_test, carries out a single trial for test on arrays
    previously transferred to GPU.

    Parameters
    ---------
    test
        As passed to gpu_trial (see).
    mats_gpu : dict of cupy arrays
        Arrays on GPU.
    """
    output = mats_gpu['output']
    if test=='dense_mult':
        # Dense-matrix multiplication.
        cp.matmul(mats_gpu['aa'],mats_gpu['bb'],out=output)
    elif test in ('drb_rn_np','drb_rn_csri','drb_rn_csrf'):
        # Calculation of bond vectors from 1D node positions...
        x = mats_gpu['x']
        if test=='drb_rn_np':
            # ...using array indexing, or...
            output[:] = x[mats_gpu['nb2']] - x[mats_gpu['nb1']]
        elif test=='drb_rn_csri':
            # ...using int CSR matrix, or..
            output[:] = mats_gpu['nbmat_csri'].dot(x)
        elif test=='drb_rn_csrf':
            # ...using float CSR matrix.
            output[:] = mats_gpu['nbmat_csrf'].dot(x)
    elif test in ('fn_fb_at','fn_fb_csrf'):
        # Calculation of node forces from 1D bond forces...
        fb = mats_gpu['fb']
        if test=='fn_fb_at':
            # ...using ufunc.at, or...
            output.fill(0.)
            cp.add.at(output,mats_gpu['nb2'],fb)
            cp.subtract.at(output,mats_gpu['nb1'],fb)
        elif test=='fn_fb_csrf':
            # ...using flot CSR matrix.
            output[:] = mats_gpu['nbmat2_csrf'].dot(fb)
    
def sample(a):
    """Returns a copy of a small sample of a 1D or 2D array."""
    if len(a.shape)==1:
        return np.copy(a[:3])
    return np.copy(a[:3,:3])
    
if __name__=='__main__':
    print(f'Running: {THIS_IS}')
    print(f'Local time: {time.asctime(time.localtime())}')
    print(gpuinfo)
    if CPU:
        print(f'CPU timings use last {TUSE} of {TSKIP+TUSE} trials')
    if gpu:
        print(f'GPU timings use last {TUSE_GPU} of'
              f' {TSKIP_GPU+TUSE_GPU} trials')
    tstart = time.perf_counter()
    for test in TESTS:
        if test in ALLTESTS:
            print(f'\n\n***************** Doing test {test} ******************'
                  f'\n{ALLTESTS[test]}\n'
                  '*********************************************************')
            unsup = UNSUPGPU.get(test) # message if test is unsupported on GPU
            if gpu and unsup:
                print(unsup)
            for fptype in FPTYPES:
                print(f'\n************ Using {fptype} **************')
                print(f'{"N":>10}'
                      f'{"flop":>9}'
                      f'{"make mats":>10}',end='')
                if CPU:
                    print(f'{"CPU test":>10}'
                          f'{"*CPU op/s*":>11}',end='')
                if gpu and not unsup:
                    print(f'{"GPU test":>10}'
                          f'{"*GPU op/s*":>11}'
                          f'{"GPU xfer":>10}'
                          f'{"xfer rate":>10}',end='')
                print()
                for nn in NS:
                    mats = get_mats(test,fptype,nn)
                    nn = mats['nn']          # actual (adjusted) nn
                    flop = mats['flop']      # FP ops for one trial of test
                    print(f'{nn:10,d}'
                          f'{flop:9.2e}'
                          f'{mats["tmats"]:9.2e}s',end='',flush=True)
                    if CPU:
                        # Do test on CPU, report average time req'd and flops.
                        t_cpu = cpu_test(test,mats)
                        sample_cpu = sample(mats['output'])
                        flops_cpu = flop/t_cpu
                        print(f'{t_cpu:9.2e}s'
                              f'{flops_cpu:9.2e}/s',end='',flush=True)
                    if gpu and not unsup:
                        # Do test on GPU, report average time req'd, flops,
                        # and CPU-GPU transfer rate.
                        t_gpu,tx = gpu_test(test,mats)
                        sample_gpu = sample(mats['output'])
                        flops_gpu = flop/t_gpu
                        xrate = mats['xbytes']/tx
                        print(f'{t_gpu:9.2e}s'
                              f'{flops_gpu:9.2e}/s'
                              f'{tx:9.2e}s'
                              f'{xrate/1e9:6.2f}GB/s',end='',flush=True)
                    print()
                # Print sample of test output from CPU and/or GPU.
                if CPU:
                    print(f'\nSample of last CPU result:\n{sample_cpu}')
                if gpu and not unsup:
                    print(f'\nSample of last GPU result:\n{sample_gpu}')
    ttotal = time.perf_counter() - tstart
    print(f'\nTotal time to run program: {ttotal:,.3f}s')

""" ******************* end of module gputest.py *********************** """