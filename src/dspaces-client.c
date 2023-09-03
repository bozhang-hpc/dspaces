/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers
 * University
 *
 * See COPYRIGHT in top-level directory.
 */
#include "dspaces.h"
#include "dspacesp.h"
#include "gspace.h"
#include "ss_data.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <dirent.h>
#include <rdma/fabric.h>

#ifdef HAVE_DRC
#include <rdmacred.h>
#endif /* HAVE_DRC */

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

#include <mpi.h>

#define DEBUG_OUT(...)                                                         \
    do {                                                                       \
        if(client->f_debug) {                                                  \
            fprintf(stderr, "Rank %i: %s, line %i (%s): ", client->rank,       \
                    __FILE__, __LINE__, __func__);                             \
            fprintf(stderr, __VA_ARGS__);                                      \
        }                                                                      \
    } while(0);

#define SUB_HASH_SIZE 16

#define CUDA_ASSERT(x)                                                          \
    do                                                                          \
        {                                                                       \
            if (!(x))                                                           \
                {                                                               \
                    fprintf(stderr, "Rank %i: %s, line %i (%s):"                \
                            "Assertion %s failed!\n",                           \
                            client->rank, __FILE__, __LINE__, __func__, #x);    \
                    return dspaces_ERR_CUDA;                                    \
                }                                                               \
        } while (0)

#define CUDA_ASSERTRT(stmt)				                                        \
    do                                                                          \
        {                                                                       \
            cudaError_t err = (stmt);                                           \
            if (err != cudaSuccess) {                                           \
                fprintf(stderr, "Rank %i: %s, line %i (%s):"                    \
                        "%s failed, Err Code: (%s)\n",                          \
                        client->rank, __FILE__, __LINE__, __func__, #stmt,      \
                        cudaGetErrorString(err));                               \
            }                                                                   \
            CUDA_ASSERT(cudaSuccess == err);                                \
        } while (0)

#define CUDA_ASSERTDRV(stmt)				                                    \
    do                                                                          \
        {                                                                       \
            CUresult result = (stmt);                                           \
            if (result != CUDA_SUCCESS) {                                       \
                const char *_err_name;                                          \
                cuGetErrorName(result, &_err_name);                             \
                fprintf(stderr, "Rank %i: %s, line %i (%s):"                    \
                        "%s failed, Err Code: (%s)\n",                          \
                        client->rank, __FILE__, __LINE__, __func__, #stmt,      \
                        _err_name);                                             \
            }                                                                   \
            CUDA_ASSERT(CUDA_SUCCESS == result);                                \
        } while (0)

#define CUDA_MEM_ALIGN(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

#define CUDA_MAX_CONCURRENT_KERNELS 128
#define DSPACES_CUDA_DEFAULT_CONCURRENT_KERNELS 32

// static int g_is_initialized = 0;

static enum storage_type st = column_major;

struct dspaces_sub_handle {
    struct dspaces_req *req;
    void *arg;
    int result;
    int status;
    int id;
    dspaces_sub_fn cb;
    obj_descriptor q_odsc;
};

struct sub_list_node {
    struct sub_list_node *next;
    struct dspaces_sub_handle *subh;
    int id;
};

struct nic_list_entry {
    struct list_head entry;
    char* name;
};

enum dspaces_cuda_dev_mode {dspaces_CUDA_PIPELINE, dspaces_CUDA_GDR, dspaces_CUDA_GDRCOPY};

struct dspaces_cuda_dev_info {
    enum dspaces_cuda_dev_mode mode;
    CUdevice dev;
    CUcontext dev_ctx;
};

struct dspaces_cuda_info {
    int cuda_put_mode;
    int cuda_get_mode;
    int visible_dev_num;
    int total_dev_num;
    int nic_num;
    int concurrency_enabled;
    int num_concurrent_kernels;
    struct dspaces_cuda_dev_info *dev_list;
#ifdef HAVE_GDRCOPY
    gdr_t gdrcopy_handle;
#endif
};


struct dspaces_put_req {
    hg_handle_t handle;
    margo_request req;
    struct dspaces_put_req *next;
    bulk_gdim_t in;
    int finalized;
    void *buffer;
    hg_return_t ret;
};

struct dspaces_client {
    margo_instance_id mid;
    hg_id_t put_id;
    hg_id_t put_local_id;
    hg_id_t put_meta_id;
    hg_id_t get_id;
    hg_id_t get_local_id;
    hg_id_t query_id;
    hg_id_t query_meta_id;
    hg_id_t ss_id;
    hg_id_t drain_id;
    hg_id_t kill_id;
    hg_id_t kill_client_id;
    hg_id_t sub_id;
    hg_id_t notify_id;
    hg_id_t put_dc_id;
    hg_id_t putlocal_subdrain_id;
    hg_id_t notify_drain_id;
    hg_id_t sub_ods_id;
    hg_id_t notify_ods_id;
    struct dc_gspace *dcg;
    char **server_address;
    char **node_names;
    char my_node_name[HOST_NAME_MAX];
    int my_server;
    int size_sp;
    int rank;
    int local_put_count; // used during finalize
    int f_debug;
    int f_final;
    struct dspaces_cuda_info cuda_info;
    int listener_init;
    struct dspaces_put_req *put_reqs;
    struct dspaces_put_req *put_reqs_end;

    int sub_serial;
    int sub_ods_serial;
    struct sub_list_node *sub_lists[SUB_HASH_SIZE];
    struct sub_list_node *done_list;
    int pending_sub;

#ifdef HAVE_DRC
    uint32_t drc_credential_id;
#endif /* HAVE_DRC */

    ABT_mutex ls_mutex;
    ABT_mutex drain_mutex;
    ABT_mutex sub_mutex;
    ABT_mutex putlocal_subdrain_mutex;
    ABT_mutex sub_ods_mutex;
    ABT_cond drain_cond;
    ABT_cond sub_cond;

    ABT_xstream listener_xs;
};

DECLARE_MARGO_RPC_HANDLER(get_local_rpc)
static void get_local_rpc(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(drain_rpc)
static void drain_rpc(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(kill_client_rpc)
static void kill_client_rpc(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(notify_rpc)
static void notify_rpc(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(notify_drain_rpc)
static void notify_drain_rpc(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(notify_ods_rpc)
static void notify_ods_rpc(hg_handle_t h);

// round robin fashion
// based on how many clients processes are connected to the server
static hg_return_t get_server_address(dspaces_client_t client,
                                      hg_addr_t *server_addr)
{
    return (margo_addr_lookup(
        client->mid, client->server_address[client->my_server], server_addr));
}

static hg_return_t get_meta_server_address(dspaces_client_t client,
                                           hg_addr_t *server_addr)
{
    return (
        margo_addr_lookup(client->mid, client->server_address[0], server_addr));
}

static void choose_server(dspaces_client_t client)
{
    int match_count = 0;
    int i;

    gethostname(client->my_node_name, HOST_NAME_MAX);

    for(i = 0; i < client->size_sp; i++) {
        if(strcmp(client->my_node_name, client->node_names[i]) == 0) {
            match_count++;
        }
    }
    if(match_count) {
        DEBUG_OUT("found %i servers that share a node with me.\n", match_count);
        match_count = client->rank % match_count;
        for(i = 0; i < client->size_sp; i++) {
            if(strcmp(client->my_node_name, client->node_names[i]) == 0) {
                if(match_count == 0) {
                    DEBUG_OUT("Attaching to server %i.\n", i);
                    client->my_server = i;
                    break;
                }
                match_count--;
            }
        }
    } else {
        client->my_server = client->rank % client->size_sp;
        DEBUG_OUT(
            "No on-node servers found. Attaching round-robin to server %i.\n",
            client->my_server);
        return;
    }
}

static int get_ss_info(dspaces_client_t client)
{
    hg_return_t hret;
    hg_handle_t handle;
    ss_information out;
    hg_addr_t server_addr;
    int ret = dspaces_SUCCESS;

    get_server_address(client, &server_addr);

    /* create handle */
    hret = margo_create(client->mid, server_addr, client->ss_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        return dspaces_ERR_MERCURY;
    }

    DEBUG_OUT("Sending ss_rpc\n");
    hret = margo_forward(handle, NULL);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s):  margo_forward() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }
    DEBUG_OUT("Got ss_rpc reply\n");
    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }
    ss_info_hdr ss_data;
    memcpy(&ss_data, out.ss_buf.raw_odsc, sizeof(ss_info_hdr));

    client->dcg->ss_info.num_dims = ss_data.num_dims;
    client->dcg->ss_info.num_space_srv = ss_data.num_space_srv;
    memcpy(&(client->dcg->ss_domain), &(ss_data.ss_domain),
           sizeof(struct bbox));
    client->dcg->max_versions = ss_data.max_versions;
    client->dcg->hash_version = ss_data.hash_version;
    memcpy(&(client->dcg->default_gdim), &(ss_data.default_gdim),
           sizeof(struct global_dimension));

    margo_free_output(handle, &out);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);
    return ret;
}

static struct dc_gspace *dcg_alloc(dspaces_client_t client)
{
    struct dc_gspace *dcg_l;

    (void)client;
    dcg_l = calloc(1, sizeof(*dcg_l));
    if(!dcg_l)
        goto err_out;

    INIT_LIST_HEAD(&dcg_l->locks_list);
    init_gdim_list(&dcg_l->gdim_list);
    dcg_l->hash_version = ssd_hash_version_v1; // set default hash versio

    // added for gpu data movement path selection
    INIT_LIST_HEAD(&dcg_l->gpu_bulk_put_list);
    INIT_LIST_HEAD(&dcg_l->gpu_bulk_get_list);

    // added for gpu dcds pending requests
    INIT_LIST_HEAD(&dcg_l->putlocal_subdrain_list);

    // added for gpu dcds get pattern records
    INIT_LIST_HEAD(&dcg_l->getobj_record_list);

    // added for gpu dcds get ods subscription
    INIT_LIST_HEAD(&dcg_l->sub_ods_list);

    return dcg_l;

err_out:
    fprintf(stderr, "'%s()': failed.\n", __func__);
    return NULL;
}

FILE *open_conf_ds(dspaces_client_t client)
{
    int wait_time, time = 0;
    FILE *fd;

    do {
        fd = fopen("conf.ds", "r");
        if(!fd) {
            if(errno == ENOENT) {
                DEBUG_OUT("unable to find config file 'conf.ds' after %d "
                          "seconds, will try again...\n",
                          time);
            } else {
                fprintf(stderr, "could not open config file 'conf.ds'.\n");
                return (NULL);
            }
        }
        wait_time = (rand() % 3) + 1;
        time += wait_time;
        sleep(wait_time);
    } while(!fd);

    return (fd);
}

static int read_conf(dspaces_client_t client, char **listen_addr_str)
{
    int size;
    FILE *fd;
    fpos_t lstart;
    int i, ret;

    ret = -1;
    fd = open_conf_ds(client);
    if(!fd) {
        goto fini;
    }

    fscanf(fd, "%d\n", &client->size_sp);
    client->server_address =
        malloc(client->size_sp * sizeof(*client->server_address));
    client->node_names = malloc(client->size_sp * sizeof(*client->node_names));
    for(i = 0; i < client->size_sp; i++) {
        fgetpos(fd, &lstart);
        fscanf(fd, "%*s%n", &size);
        client->node_names[i] = malloc(size + 1);
        fscanf(fd, "%*s%n\n", &size);
        client->server_address[i] = malloc(size + 1);
        fsetpos(fd, &lstart);
        fscanf(fd, "%s %s\n", client->node_names[i], client->server_address[i]);
    }
    fgetpos(fd, &lstart);
    fscanf(fd, "%*s%n\n", &size);
    fsetpos(fd, &lstart);
    *listen_addr_str = malloc(size + 1);
    fscanf(fd, "%s\n", *listen_addr_str);

#ifdef HAVE_DRC
    fgetpos(fd, &lstart);
    fscanf(fd, "%" SCNu32, &client->drc_credential_id);
#endif
    fclose(fd);

    ret = 0;

fini:
    return ret;
}

static int read_conf_mpi(dspaces_client_t client, MPI_Comm comm,
                         char **listen_addr_str)
{
    FILE *fd, *conf;
    struct stat st;
    char *file_buf;
    int file_len;
    int rank;
    int size;
    fpos_t lstart;
    int i;

    MPI_Comm_rank(comm, &rank);
    if(rank == 0) {
        fd = open_conf_ds(client);
        if(fd == NULL) {
            file_len = -1;
        } else {
            fstat(fileno(fd), &st);
            file_len = st.st_size;
        }
    }
    MPI_Bcast(&file_len, 1, MPI_INT, 0, comm);
    if(file_len == -1) {
        return (-1);
    }
    file_buf = malloc(file_len);
    if(rank == 0) {
        fread(file_buf, 1, file_len, fd);
        fclose(fd);
    }
    MPI_Bcast(file_buf, file_len, MPI_BYTE, 0, comm);

    conf = fmemopen(file_buf, file_len, "r");
    fscanf(conf, "%d\n", &client->size_sp);
    client->server_address =
        malloc(client->size_sp * sizeof(*client->server_address));
    client->node_names = malloc(client->size_sp * sizeof(*client->node_names));
    for(i = 0; i < client->size_sp; i++) {
        fgetpos(conf, &lstart);
        fgetpos(conf, &lstart);
        fscanf(conf, "%*s%n", &size);
        client->node_names[i] = malloc(size + 1);
        fscanf(conf, "%*s%n\n", &size);
        client->server_address[i] = malloc(size + 1);
        fsetpos(conf, &lstart);
        fscanf(conf, "%s %s\n", client->node_names[i],
               client->server_address[i]);
    }
    fgetpos(conf, &lstart);
    fscanf(conf, "%*s%n\n", &size);
    fsetpos(conf, &lstart);
    *listen_addr_str = malloc(size + 1);
    fscanf(conf, "%s\n", *listen_addr_str);

#ifdef HAVE_DRC
    fgetpos(conf, &lstart);
    fscanf(conf, "%" SCNu32, &client->drc_credential_id);
#endif

    fclose(conf);
    free(file_buf);

    return (0);
}

static int dspaces_init_internal(int rank, dspaces_client_t *c)
{
    const char *envdebug = getenv("DSPACES_DEBUG");
    static int is_initialized = 0;

    if(is_initialized) {
        fprintf(stderr,
                "DATASPACES: WARNING: %s: multiple instantiations of the "
                "dataspaces client are not supported.\n",
                __func__);
        return (dspaces_ERR_ALLOCATION);
    }
    dspaces_client_t client = (dspaces_client_t)calloc(1, sizeof(*client));
    if(!client)
        return dspaces_ERR_ALLOCATION;

    if(envdebug) {
        client->f_debug = 1;
    }

    client->rank = rank;
    client->put_reqs = NULL;
    client->put_reqs_end = NULL;

    // now do dcg_alloc and store gid
    client->dcg = dcg_alloc(client);

    if(!(client->dcg))
        return dspaces_ERR_ALLOCATION;

    is_initialized = 1;

    *c = client;

    return dspaces_SUCCESS;
}

#ifdef HAVE_GDRCOPY
static int check_gdrcopy_support_dev(dspaces_client_t client, int dev_rank)
{
    int gdr_support;
    int cur_dev_rank;

    CUDA_ASSERTRT(cudaGetDevice(&cur_dev_rank));
    CUDA_ASSERTRT(cudaSetDevice(dev_rank));

    #if CUDA_VERSION >= 11030
    int drv_version;
    CUDA_ASSERTDRV(cuDriverGetVersion(&drv_version));
    // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA support.
    if(drv_version >= 11030) {
        CUdevice dev;
        CUDA_ASSERTDRV(cuDeviceGet(&dev, dev_rank));
        CUDA_ASSERTDRV(cuDeviceGetAttribute(&gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));
        CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));
        return gdr_support;
    }
    #endif

    // For older versions, we fall back to detect this support with gdr_pin_buffer.
    const size_t buf_size = GPU_PAGE_SIZE;
    CUdeviceptr d_A;
    
    // Malloc page-aligned memory on device
    CUdeviceptr ptr;
    size_t allocated_size;
    allocated_size = buf_size + GPU_PAGE_SIZE - 1;
    CUDA_ASSERTDRV(cuMemAlloc(&ptr, allocated_size));

    // Don't need sync control since we are just checking gdrcopy support

    d_A = CUDA_MEM_ALIGN(ptr, GPU_PAGE_SIZE);

    // gdr_t gdr = gdr_open();
    // if(!gdr) {
    //     // fprintf(stderr, "ERROR: (%s): gdr_open() failed, is gdrdrv driver installed and loaded?\n", __func__);
    //     gdr_support = 0;
    //     CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));
    //     return gdr_support;
    // }

    gdr_mh_t gdrmh;
    int gdrret;
    gdrret = gdr_pin_buffer(client->cuda_info.gdrcopy_handle, d_A, buf_size, 0, 0, &gdrmh);
    if(gdrret != 0) {
        // fprintf(stderr, "ERROR: (%s): gdr_pin_buffer() failed, Err Code: (%d)\n"
        //                 "GPU %d might not support GPUDirect RDMA\n", __func__, gdrret, data_dev_rank);
        gdr_support = 0;
        CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));
        return gdr_support;
    }

    gdrret = gdr_unpin_buffer(client->cuda_info.gdrcopy_handle, gdrmh);
    if(gdrret != 0) {
        // fprintf(stderr, "ERROR: (%s): gdr_unpin_buffer() failed, Err Code: (%d)\n", __func__, gdrret);
        gdr_support = 0;
        CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));
        return gdr_support;
    }

    // gdrret = gdr_close(gdr);
    // if(gdrret != 0) {
    //     // fprintf(stderr, "ERROR: (%s): gdr_close() failed, Err Code: (%d)\n", __func__, gdrret);
    //     gdr_support = 0;
    //     CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));
    //     return gdr_support;
    // }

    CUDA_ASSERTDRV(cuMemFree(ptr));

    gdr_support = 1;
    CUDA_ASSERTRT(cudaSetDevice(cur_dev_rank));

    return gdr_support;
}

static int check_gdrcopy_support_dev_all(dspaces_client_t client)
{
    for(int dev_rank=0; dev_rank<client->cuda_info.visible_dev_num; dev_rank++) {
        int gdr_support = check_gdrcopy_support_dev(client, dev_rank);
        if(gdr_support != dspaces_ERR_CUDA) {
            // add info to cuda device table
            if(gdr_support) {
                client->cuda_info.dev_list[dev_rank].mode = dspaces_CUDA_GDRCOPY;
            }
        } else{
            return dspaces_ERR_CUDA;
        }
    }

    return dspaces_SUCCESS;
}

static int gdrcopy_init(dspaces_client_t client)
{
    for(int dev_rank=0; dev_rank<client->cuda_info.visible_dev_num; dev_rank++) {
        CUDA_ASSERTDRV(cuDevicePrimaryCtxRetain(&(client->cuda_info.dev_list[dev_rank].dev_ctx),
                                                client->cuda_info.dev_list[dev_rank].dev));
    }
    client->cuda_info.gdrcopy_handle = gdr_open();
    if(!client->cuda_info.gdrcopy_handle) {
        fprintf(stderr, "Error: Rank %i: gdr_open() failed!\n", client->rank);
        return dspaces_ERR_GDRCOPY;
    }
    return dspaces_SUCCESS;
}

static inline void gdrcopy_fini(dspaces_client_t client)
{
    gdr_close(client->cuda_info.gdrcopy_handle);
}
#endif

static int dspaces_init_gpu(dspaces_client_t client)
{
    int ret = dspaces_SUCCESS;

    const char* envcudaconcurrent = getenv("DSPACES_CUDA_ENABLE_CONCURRENCY");
    const char* envcudaconcurrentkernels = getenv("DSPACES_CUDA_NUM_CONCURRENT_KERNELS");

    if(envcudaconcurrent) {
        client->cuda_info.concurrency_enabled = 1;
    } else {
        client->cuda_info.concurrency_enabled = 0;
    }

    // TODO: set the concurrent kernel nums according to different devices
    if(envcudaconcurrentkernels) {
        int concurrent_kernels = atoi(envcudaconcurrentkernels);
        if(concurrent_kernels < CUDA_MAX_CONCURRENT_KERNELS) {
            client->cuda_info.num_concurrent_kernels = concurrent_kernels;
        } else {
            client->cuda_info.num_concurrent_kernels = DSPACES_CUDA_DEFAULT_CONCURRENT_KERNELS;
        }
    } else {
        client->cuda_info.num_concurrent_kernels = DSPACES_CUDA_DEFAULT_CONCURRENT_KERNELS;
    }

    const char *envcudaputmode = getenv("DSPACES_CUDA_PUT_MODE");
    const char *envcudagetmode = getenv("DSPACES_CUDA_GET_MODE");

    // Default Put Mode: 0 - Hybrid, Others: 1 - Baseline, 2 - Pipeline, 3 - GDR, 4 - GDRCopy
    // 5 - Heuristic, 6 - Dual Channel, 7 - Dual Channel Dual Staging
    if(envcudaputmode) {
        int cudaputmode = atoi(envcudaputmode);
        // mode check 0-7
        if(cudaputmode >= 0 && cudaputmode < 9) {
            client->cuda_info.cuda_put_mode = cudaputmode;
        } else {
            client->cuda_info.cuda_put_mode = 0;
        }
    } else {
        client->cuda_info.cuda_put_mode = 0;
    }

    // Default Get Mode: 1 - Baseline, Others: 2 - GDR, 3 - Hybrid, 4 - Heuristic
    // 5 - Dual Channel, 6 - Daul Channel Dual Staging
    if(envcudagetmode) {
        int cudagetmode = atoi(envcudagetmode);

        // mode check 1-6
        if(cudagetmode >= 1 && cudagetmode < 7) {
            client->cuda_info.cuda_get_mode = cudagetmode;
        } else {
            client->cuda_info.cuda_get_mode = 1;
        }
    } else {
        client->cuda_info.cuda_get_mode = 1;
    }

    CUDA_ASSERTRT(cudaGetDeviceCount(&client->cuda_info.visible_dev_num));
    client->cuda_info.dev_list = (struct dspaces_cuda_dev_info*) malloc(client->cuda_info.visible_dev_num*sizeof(struct dspaces_cuda_dev_info));

    // Get Device Info
    for(int dev_rank=0; dev_rank<client->cuda_info.visible_dev_num; dev_rank++) {
        CUDA_ASSERTDRV(cuDeviceGet(&(client->cuda_info.dev_list[dev_rank].dev), dev_rank));
    }

#ifdef HAVE_GDRCOPY
    if(client->cuda_info.cuda_put_mode == 4) {
        ret = gdrcopy_init(client);
        if(ret == dspaces_SUCCESS) {
            ret = check_gdrcopy_support_dev_all(client);
            if(ret != dspaces_SUCCESS) {
                gdrcopy_fini(client);
                fprintf(stderr, "Error: Rank %i: check_gdrcopy_support_dev_all() failed!\n", client->rank);
                return dspaces_ERR_CUDA;
            }
        } else {
            fprintf(stderr, "Error: Rank %i: gdrcopy_init() failed!\n", client->rank);
            return dspaces_ERR_GDRCOPY;
        }
    }
#endif

    char hint[32];
    switch (client->cuda_info.cuda_put_mode)
    {
    case 0:
        sprintf(hint, "Hybrid");
        break;
    case 1:
        sprintf(hint, "Baseline");
        break;    
    case 2:
        sprintf(hint, "Pipeline");
        break;
    case 3:
        sprintf(hint, "GDR");
        break;
    case 4:
        sprintf(hint, "GDRCopy");
        break;
    case 5:
        sprintf(hint, "Heuristic");
        break;
    case 6:
        sprintf(hint, "Dual Channel");
        break;
    case 7:
        sprintf(hint, "Dual Channel Dual Staging");
        break;
    case 8:
        sprintf(hint, "Dual Channel Dual Staging V2");
        break;
    default:
        sprintf(hint, "Error");
        break;
    }

    DEBUG_OUT("dspaces CUDA Put Mode = %s\n", hint);

    switch (client->cuda_info.cuda_get_mode)
    {
    case 1:
        sprintf(hint, "Baseline");
        break;  
    case 2:
        sprintf(hint, "GDR");
        break;
    case 3:
        sprintf(hint, "Hybrid");
        break;
    case 4:
        sprintf(hint, "Heuristic");
        break;
    case 5:
        sprintf(hint, "Dual Channel");
        break;
    case 6:
        sprintf(hint, "Dual Channel Dual Staging");
        break;
    default:
        sprintf(hint, "Error");
        break;
    }

    DEBUG_OUT("dspaces CUDA Get Mode = %s\n", hint);

    return dspaces_SUCCESS;
}

static int dspaces_init_margo(dspaces_client_t client,
                              const char *listen_addr_str)
{
    hg_class_t *hg;
    struct hg_init_info hii = HG_INIT_INFO_INITIALIZER;
    char margo_conf[1024];
    struct margo_init_info mii = MARGO_INIT_INFO_INITIALIZER;
    int i;

    margo_set_environment(NULL);
    sprintf(margo_conf,
            "{ \"use_progress_thread\" : false, \"rpc_thread_count\" : 0}");
    hii.request_post_init = 1024;
    hii.auto_sm = false;
    hii.no_multi_recv = true;
    if(client->cuda_info.cuda_put_mode == 1 && client->cuda_info.cuda_get_mode != 2) {
        hii.no_bulk_eager=0;
        hii.na_init_info.request_mem_device = false;
    } else {
        hii.no_bulk_eager=1;
        hii.na_init_info.request_mem_device = true;
    }
    mii.hg_init_info = &hii;
    mii.json_config = margo_conf;
    ABT_init(0, NULL);

#ifdef HAVE_DRC
    int ret = 0;
    drc_info_handle_t drc_credential_info;
    uint32_t drc_cookie;
    char drc_key_str[256] = {0};

    ret = drc_access(client->drc_credential_id, 0, &drc_credential_info);
    if(ret != DRC_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): drc_access failure %d\n", __func__, ret);
        return ret;
    }

    drc_cookie = drc_get_first_cookie(drc_credential_info);
    sprintf(drc_key_str, "%u", drc_cookie);

    hii.na_init_info.auth_key = drc_key_str;

    client->mid =
        margo_init_ext(listen_addr_str, MARGO_SERVER_MODE, &mii);

#else
    client->mid = margo_init_ext(listen_addr_str, MARGO_SERVER_MODE, &mii);
    if(client->f_debug) {
        if(!client->rank) {
            char *margo_json = margo_get_config(client->mid);
            fprintf(stderr, "%s", margo_json);
            free(margo_json);
        }
        margo_set_log_level(client->mid, MARGO_LOG_TRACE);
    }

#endif /* HAVE_DRC */

    if(!client->mid) {
        fprintf(stderr, "ERROR: %s: margo_init() failed.\n", __func__);
        return (dspaces_ERR_MERCURY);
    }

    hg = margo_get_class(client->mid);

    ABT_mutex_create(&client->ls_mutex);
    ABT_mutex_create(&client->drain_mutex);
    ABT_mutex_create(&client->sub_mutex);
    ABT_mutex_create(&client->putlocal_subdrain_mutex);
    ABT_mutex_create(&client->sub_ods_mutex);
    ABT_cond_create(&client->drain_cond);
    ABT_cond_create(&client->sub_cond);

    for(i = 0; i < SUB_HASH_SIZE; i++) {
        client->sub_lists[i] = NULL;
    }
    client->done_list = NULL;
    client->sub_serial = 0;
    client->sub_ods_serial = 0;
    client->pending_sub = 0;

    /* check if RPCs have already been registered */
    hg_bool_t flag;
    hg_id_t id;
    margo_registered_name(client->mid, "put_rpc", &id, &flag);

    if(flag == HG_TRUE) { /* RPCs already registered */
        margo_registered_name(client->mid, "put_rpc", &client->put_id, &flag);
        margo_registered_name(client->mid, "put_local_rpc",
                              &client->put_local_id, &flag);
        margo_registered_name(client->mid, "put_meta_rpc", &client->put_meta_id,
                              &flag);
        margo_registered_name(client->mid, "get_rpc", &client->get_id, &flag);
        margo_registered_name(client->mid, "get_local_rpc",
                              &client->get_local_id, &flag);
        DS_HG_REGISTER(hg, client->get_local_id, bulk_in_t, bulk_out_t,
                       get_local_rpc);
        margo_registered_name(client->mid, "query_rpc", &client->query_id,
                              &flag);
        margo_registered_name(client->mid, "ss_rpc", &client->ss_id, &flag);
        margo_registered_name(client->mid, "drain_rpc", &client->drain_id,
                              &flag);
        DS_HG_REGISTER(hg, client->drain_id, bulk_in_t, bulk_out_t, drain_rpc);
        margo_registered_name(client->mid, "kill_rpc", &client->kill_id, &flag);
        margo_registered_name(client->mid, "kill_client_rpc",
                              &client->kill_client_id, &flag);
        DS_HG_REGISTER(hg, client->kill_client_id, int32_t, void,
                       kill_client_rpc);
        margo_registered_name(client->mid, "sub_rpc", &client->sub_id, &flag);
        margo_registered_name(client->mid, "notify_rpc", &client->notify_id,
                              &flag);
        DS_HG_REGISTER(hg, client->notify_id, odsc_list_t, void, notify_rpc);
        margo_registered_name(client->mid, "query_meta_rpc",
                              &client->query_meta_id, &flag);
        margo_registered_name(client->mid, "put_dc_rpc", &client->put_dc_id, &flag);
        margo_registered_name(client->mid, "putlocal_subdrain_rpc",
                                &client->putlocal_subdrain_id, &flag);
        margo_registered_name(client->mid, "notify_drain_rpc",
                              &client->notify_drain_id, &flag);
        DS_HG_REGISTER(hg, client->notify_drain_id, odsc_list_t, void, notify_drain_rpc);
        margo_registered_name(client->mid, "sub_ods_rpc", &client->sub_ods_id, &flag);
        margo_registered_name(client->mid, "notify_ods_rpc", &client->notify_ods_id,
                              &flag);
        DS_HG_REGISTER(hg, client->notify_ods_id, odsc_list_t, void, notify_ods_rpc);
    } else {
        client->put_id = MARGO_REGISTER(client->mid, "put_rpc", bulk_gdim_t,
                                        bulk_out_t, NULL);
        client->put_local_id = MARGO_REGISTER(client->mid, "put_local_rpc",
                                              odsc_gdim_t, bulk_out_t, NULL);
        client->put_meta_id = MARGO_REGISTER(client->mid, "put_meta_rpc",
                                             put_meta_in_t, bulk_out_t, NULL);
        margo_register_data(client->mid, client->put_meta_id, (void *)client,
                            NULL);
        client->get_id =
            MARGO_REGISTER(client->mid, "get_rpc", bulk_in_t, bulk_out_t, NULL);
        margo_register_data(client->mid, client->get_id, (void *)client, NULL);
        client->get_local_id = MARGO_REGISTER(
            client->mid, "get_local_rpc", bulk_in_t, bulk_out_t, get_local_rpc);
        margo_register_data(client->mid, client->get_local_id, (void *)client,
                            NULL);
        client->query_id = MARGO_REGISTER(client->mid, "query_rpc", odsc_gdim_t,
                                          odsc_list_t, NULL);
        client->query_meta_id =
            MARGO_REGISTER(client->mid, "query_meta_rpc", query_meta_in_t,
                           query_meta_out_t, NULL);
        client->ss_id =
            MARGO_REGISTER(client->mid, "ss_rpc", void, ss_information, NULL);
        client->drain_id = MARGO_REGISTER(client->mid, "drain_rpc", bulk_in_t,
                                          bulk_out_t, drain_rpc);
        margo_register_data(client->mid, client->drain_id, (void *)client,
                            NULL);
        client->kill_id =
            MARGO_REGISTER(client->mid, "kill_rpc", int32_t, void, NULL);
        margo_registered_disable_response(client->mid, client->kill_id,
                                          HG_TRUE);
        margo_register_data(client->mid, client->kill_id, (void *)client, NULL);
        client->kill_client_id = MARGO_REGISTER(client->mid, "kill_client_rpc",
                                                int32_t, void, kill_client_rpc);
        margo_registered_disable_response(client->mid, client->kill_client_id,
                                          HG_TRUE);
        margo_register_data(client->mid, client->kill_client_id, (void *)client,
                            NULL);
        client->sub_id =
            MARGO_REGISTER(client->mid, "sub_rpc", odsc_gdim_t, void, NULL);
        margo_registered_disable_response(client->mid, client->sub_id, HG_TRUE);
        client->notify_id = MARGO_REGISTER(client->mid, "notify_rpc",
                                           odsc_list_t, void, notify_rpc);
        margo_register_data(client->mid, client->notify_id, (void *)client,
                            NULL);
        margo_registered_disable_response(client->mid, client->notify_id,
                                          HG_TRUE);
        client->put_dc_id = MARGO_REGISTER(client->mid, "put_dc_rpc", dc_bulk_gdim_t,
                                        bulk_out_t, NULL);
        client->putlocal_subdrain_id =
            MARGO_REGISTER(client->mid, "putlocal_subdrain_rpc", bulk_gdim_t,
                                        bulk_out_t, NULL);
        client->notify_drain_id = MARGO_REGISTER(client->mid, "notify_drain_rpc",
                                           odsc_list_t, void, notify_drain_rpc);
        margo_register_data(client->mid, client->notify_drain_id, (void *)client,
                            NULL);
        margo_registered_disable_response(client->mid, client->notify_drain_id,
                                          HG_TRUE);
        client->sub_ods_id =
            MARGO_REGISTER(client->mid, "sub_ods_rpc", odsc_gdim_t, void, NULL);
        margo_registered_disable_response(client->mid, client->sub_ods_id, HG_TRUE);
        client->notify_ods_id = MARGO_REGISTER(client->mid, "notify_ods_rpc",
                                           odsc_list_t, void, notify_ods_rpc);
        margo_register_data(client->mid, client->notify_ods_id, (void *)client,
                            NULL);
        margo_registered_disable_response(client->mid, client->notify_ods_id,
                                          HG_TRUE);
    }

    return (dspaces_SUCCESS);
}

static int read_topology_mpi(dspaces_client_t client, MPI_Comm comm, char* listen_addr_str)
{
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    int shmrank;
    MPI_Comm_rank(shmcomm, &shmrank);

    DIR* dirp;
    struct dirent *dire;
    struct fi_info *hints, *info, *cur;
    struct list_head nic_list;
    struct nic_list_entry *nic, *e, *t;
    client->cuda_info.total_dev_num = 0;
    int found;
    client->cuda_info.nic_num=0;
    if(shmrank == 0) {
        /* Check num of GPUs on the node
        regardless how the CUDA_VISIABLE_DEVICE is set */
        dirp = opendir("/proc/driver/nvidia/gpus");
        if(!dirp) {
            // TODO: handle error
        } else{
            while((dire = readdir(dirp)) != NULL) {
                if(strcmp(dire->d_name, ".") == 0 ||strcmp(dire->d_name, "..") == 0) {
                    continue;
                }
                client->cuda_info.total_dev_num++;
            }
        }
        /* Check num of NICs on the node according to the protocol */
        INIT_LIST_HEAD(&nic_list);
        hints = fi_allocinfo();
        hints->mode = ~0;
        hints->domain_attr->mode = ~0;
	    hints->domain_attr->mr_mode = ~(FI_MR_BASIC | FI_MR_SCALABLE);
        free(hints->fabric_attr->prov_name);
		hints->fabric_attr->prov_name = strdup(listen_addr_str);
        fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                    NULL, NULL, 0, hints, &info);
        for (cur = info; cur; cur = cur->next) {
            found = 0;
            list_for_each_entry(nic, &nic_list, struct nic_list_entry, entry) {
                if(strcmp(nic->name, cur->nic->device_attr->name)==0) {
                    found = 1;
                }
            }
            if(!found) {
                e = (struct nic_list_entry*) malloc(sizeof(*e));
                e->name = strdup(cur->nic->device_attr->name);
                list_add(&e->entry, &nic_list);
                client->cuda_info.nic_num++;
            }
        }

        list_for_each_entry_safe(nic, t, &nic_list, struct nic_list_entry, entry) {
            list_del(&nic->entry);
            free(nic);
        }
        DEBUG_OUT("Total GPU NUM = %d, NIC NUM = %d per Node\n", client->cuda_info.total_dev_num,
                    client->cuda_info.nic_num)
    }
    MPI_Bcast(&client->cuda_info.total_dev_num, 1, MPI_INT, 0, shmcomm);
    MPI_Bcast(&client->cuda_info.nic_num, 1, MPI_INT, 0, shmcomm);
    if(client->cuda_info.total_dev_num == 0 || client->cuda_info.nic_num == 0) {
        fprintf(stdout, "Warning: No CUDA GPU or NIC detected!\n");
    }
    return (dspaces_SUCCESS);
}

static int dspaces_post_init(dspaces_client_t client)
{
    choose_server(client);

    get_ss_info(client);
    DEBUG_OUT("Total max versions on the client side is %d\n",
              client->dcg->max_versions);

    client->dcg->ls = ls_alloc(client->dcg->max_versions);
    client->local_put_count = 0;
    client->f_final = 0;

    int device, totdevice;
    CUDA_ASSERTRT(cudaGetDevice(&device));
    // CUDA_ASSERTRT(cudaGetDeviceCount(&totdevice));
    meminfo_t meminfo = parse_meminfo();
    size_t d_free, d_total;
    CUDA_ASSERTRT(cudaMemGetInfo(&d_free, &d_total));
    DEBUG_OUT("Rank %d: Device = %d/%d, Host Free Memory = %lld, Device Free Memory = %zu \n",
                client->rank, device, client->cuda_info.visible_dev_num,
                meminfo.MemAvailableMiB, d_free);

    return (dspaces_SUCCESS);
}

int dspaces_init(int rank, dspaces_client_t *c)
{
    dspaces_client_t client;
    char *listen_addr_str;
    int ret;

    ret = dspaces_init_internal(rank, &client);
    if(ret != dspaces_SUCCESS) {
        return (ret);
    }

    ret = dspaces_init_gpu(client);
    if(ret != dspaces_SUCCESS) {
        return (ret);
    }

    ret = read_conf(client, &listen_addr_str);
    if(ret != 0) {
        return (ret);
    }

    dspaces_init_margo(client, listen_addr_str);

    free(listen_addr_str);

    dspaces_post_init(client);

    *c = client;

    return dspaces_SUCCESS;
}

int dspaces_init_mpi(MPI_Comm comm, dspaces_client_t *c)
{
    dspaces_client_t client;
    int rank;
    char *listen_addr_str;
    int ret;

    MPI_Comm_rank(comm, &rank);

    ret = dspaces_init_internal(rank, &client);
    if(ret != dspaces_SUCCESS) {
        return (ret);
    }

    ret = dspaces_init_gpu(client);
    if(ret != dspaces_SUCCESS) {
        return (ret);
    }

    ret = read_conf_mpi(client, comm, &listen_addr_str);
    if(ret != 0) {
        return (ret);
    }
    dspaces_init_margo(client, listen_addr_str);
    read_topology_mpi(client, comm, listen_addr_str);
    free(listen_addr_str);

    dspaces_post_init(client);

    *c = client;

    return (dspaces_SUCCESS);
}

static void free_done_list(dspaces_client_t client)
{
    struct sub_list_node *node;

    while(client->done_list) {
        node = client->done_list;
        client->done_list = node->next;
        free(node->subh);
        free(node);
    }
}

int dspaces_fini(dspaces_client_t client)
{
    DEBUG_OUT("finalizing.\n");

    ABT_mutex_lock(client->sub_mutex);
    while(client->pending_sub > 0) {
        DEBUG_OUT("Pending subscriptions: %d\n", client->pending_sub);
        ABT_cond_wait(client->sub_cond, client->sub_mutex);
    }
    ABT_mutex_unlock(client->sub_mutex);

    free_done_list(client);

    do { // watch out for spurious wake
        ABT_mutex_lock(client->drain_mutex);
        client->f_final = 1;

        if(client->local_put_count > 0) {
            DEBUG_OUT("waiting for pending drainage. %d object remain.\n",
                      client->local_put_count);
            ABT_cond_wait(client->drain_cond, client->drain_mutex);
            DEBUG_OUT("received drainage signal.\n");
        }
        ABT_mutex_unlock(client->drain_mutex);
    } while(client->local_put_count > 0);

    while(client->put_reqs) {
        dspaces_check_put(client, client->put_reqs, 1);
    }

    DEBUG_OUT("all objects drained. Finalizing...\n");

    free_gdim_list(&client->dcg->gdim_list);
    free_gpu_bulk_list(&client->dcg->gpu_bulk_put_list);
    free_gpu_bulk_list(&client->dcg->gpu_bulk_get_list);
    free_putlocal_subdrain_list(&client->dcg->putlocal_subdrain_list);
    free_getobj_list(&client->dcg->getobj_record_list);
    free_subods_list(&client->dcg->sub_ods_list);
    free(client->server_address[0]);
    free(client->server_address);
    ls_free(client->dcg->ls);
    free(client->dcg);

#ifdef HAVE_GDRCOPY
    if(client->cuda_info.cuda_put_mode == 4) {
        gdrcopy_fini(client);
        for(int dev_rank=0; dev_rank<client->cuda_info.visible_dev_num; dev_rank++) {
            CUDA_ASSERTDRV(cuDevicePrimaryCtxRelease(client->cuda_info.dev_list[dev_rank].dev));
        }
    }
#endif
    free(client->cuda_info.dev_list);

    ABT_mutex_free(&client->ls_mutex);
    ABT_mutex_free(&client->drain_mutex);
    ABT_mutex_free(&client->sub_mutex);
    ABT_mutex_free(&client->putlocal_subdrain_mutex);
    ABT_mutex_free(&client->sub_ods_mutex);
    ABT_cond_free(&client->drain_cond);
    ABT_cond_free(&client->sub_cond);

    margo_finalize(client->mid);

    free(client);

    return dspaces_SUCCESS;
}

void dspaces_define_gdim(dspaces_client_t client, const char *var_name,
                         int ndim, uint64_t *gdim)
{
    if(ndim > BBOX_MAX_NDIM) {
        fprintf(stderr, "ERROR: %s: maximum object dimensionality is %d\n",
                __func__, BBOX_MAX_NDIM);
    } else {
        update_gdim_list(&(client->dcg->gdim_list), var_name, ndim, gdim);
    }
}

static int setup_put(dspaces_client_t client, const char *var_name,
                     unsigned int ver, int elem_size, int ndim, uint64_t *lb,
                     uint64_t *ub, const void *data, hg_addr_t *server_addr,
                     hg_handle_t *handle, bulk_gdim_t *in)
{
    hg_return_t hret;
    int ret = dspaces_SUCCESS;

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in->odsc.size = sizeof(odsc);
    in->odsc.raw_odsc = (char *)(&odsc);
    in->odsc.gdim_size = sizeof(struct global_dimension);
    in->odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));
    // int *a = NULL;
    // int b = *a;
    hret = margo_bulk_create(client->mid, 1, (void **)&data, &rdma_size,
                             HG_BULK_READ_ONLY, &in->handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed\n", __func__);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, server_addr);
    /* create handle */
    hret = margo_create(client->mid, *server_addr, client->put_id, handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        margo_bulk_free(in->handle);
        return dspaces_ERR_MERCURY;
    }
}

static int dspaces_init_listener(dspaces_client_t client)
{

    ABT_pool margo_pool;
    hg_return_t hret;
    int ret = dspaces_SUCCESS;

    hret = margo_get_handler_pool(client->mid, &margo_pool);
    if(hret != HG_SUCCESS || margo_pool == ABT_POOL_NULL) {
        fprintf(stderr, "ERROR: %s: could not get handler pool (%d).\n",
                __func__, hret);
        return (dspaces_ERR_ARGOBOTS);
    }
    client->listener_xs = ABT_XSTREAM_NULL;
    ret = ABT_xstream_create_basic(ABT_SCHED_BASIC_WAIT, 1, &margo_pool,
                                   ABT_SCHED_CONFIG_NULL, &client->listener_xs);
    if(ret != ABT_SUCCESS) {
        char err_str[1000];
        ABT_error_get_str(ret, err_str, NULL);
        fprintf(stderr, "ERROR: %s: could not launch handler thread: %s\n",
                __func__, err_str);
        return (dspaces_ERR_ARGOBOTS);
    }

    client->listener_init = 1;

    return (ret);
}


int dspaces_cpu_put(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));

    hret = margo_bulk_create(client->mid, 1, (void **)&data, &rdma_size,
                             HG_BULK_READ_ONLY, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);
    return ret;
}

static int cuda_put_baseline(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data, double* itime)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    void* buffer = (void*) malloc(rdma_size);

    cudaError_t curet;
    curet = cudaMemcpy(buffer, data, rdma_size, cudaMemcpyDeviceToHost);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaMemcpy() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        free(buffer);
        return dspaces_ERR_CUDA;
    }

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));

    hret = margo_bulk_create(client->mid, 1, (void **)&buffer, &rdma_size,
                             HG_BULK_READ_ONLY, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    free(buffer);

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);
    return ret;
}

static int cuda_put_pipeline(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data, double* itime)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;
    struct timeval start, end;

    gettimeofday(&start, NULL);

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreate(&stream));
    
    void* buffer = (void*) malloc(rdma_size);

    cudaError_t curet;
    curet = cudaMemcpyAsync(buffer, data, rdma_size, cudaMemcpyDeviceToHost, stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_CUDA;
    }

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';
    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t hg_rdma_size = rdma_size;

    hret = margo_bulk_create(client->mid, 1, (void **)&buffer, &hg_rdma_size,
                             HG_BULK_READ_ONLY, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_MERCURY;
    }

    curet = cudaStreamSynchronize(stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        margo_bulk_free(in.handle);
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_CUDA;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        cudaStreamDestroy(stream);
        free(buffer);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);

    CUDA_ASSERTRT(cudaStreamDestroy(stream));
    free(buffer);

    *itime = 0;
    return ret;
}

static int cuda_put_gdr(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data)
{
    // fprintf(stdout, "cuda_put_gdr()\n");

    struct timeval start, end;
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));

    struct cudaPointerAttributes ptr_attr;
    CUDA_ASSERTRT(cudaPointerGetAttributes(&ptr_attr, data));
    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                     .device = ptr_attr.device };

    hret = margo_bulk_create_attr(client->mid, 1, (void **)&data, &rdma_size,
                                  HG_BULK_READ_ONLY, &bulk_attr, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);

    return ret;
}

#ifdef HAVE_GDRCOPY
static int cuda_put_gdrcopy(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data)
{
    // fprintf(stdout, "cuda_put_gdrcopy()\n");
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;
    int gdrret;
    gdr_mh_t gdr_mh;
    gdr_info_t gdr_info;
    CUdeviceptr d_ptr = (CUdeviceptr) data;

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    void *buffer = (void*) malloc(rdma_size);

    /* gdr_copy starts */

    gdrret = gdr_pin_buffer(client->cuda_info.gdrcopy_handle, d_ptr, rdma_size, 0, 0, &gdr_mh);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_pin_buffer() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);                 
        return dspaces_ERR_GDRCOPY;
    }

    void *map_d_ptr  = NULL;
    gdrret = gdr_map(client->cuda_info.gdrcopy_handle, gdr_mh, &map_d_ptr, rdma_size);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_map() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);
        return dspaces_ERR_GDRCOPY;
    }

    gdrret = gdr_get_info(client->cuda_info.gdrcopy_handle, gdr_mh, &gdr_info);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_get_info() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);
        return dspaces_ERR_GDRCOPY;
    }

    // remember that mappings start on a 64KB boundary, so let's
    // calculate the offset from the head of the mapping to the
    // beginning of the buffer
    int gdr_offset = gdr_info.va - d_ptr;
    void *gdr_buf_ptr = (void *)((char *)map_d_ptr + gdr_offset);

    gdrret = gdr_copy_from_mapping(gdr_mh, buffer, gdr_buf_ptr, rdma_size);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_copy_from_mapping() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);
        return dspaces_ERR_GDRCOPY;
    }

    gdrret = gdr_unmap(client->cuda_info.gdrcopy_handle, gdr_mh, map_d_ptr, rdma_size);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_unmap() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);
        return dspaces_ERR_GDRCOPY;
    }

    gdrret = gdr_unpin_buffer(client->cuda_info.gdrcopy_handle, gdr_mh);
    if(gdrret != 0) {
        fprintf(stderr, "Rank %i: %s, line %i (%s): gdr_unpin_buffer() failed!, Err Code = %i",
                         client->rank, __FILE__, __LINE__, __func__, gdrret);
        free(buffer);
        return dspaces_ERR_GDRCOPY;
    }

    /* gdr_copy ends*/

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));

    hret = margo_bulk_create(client->mid, 1, (void **)&buffer, &rdma_size,
                             HG_BULK_READ_ONLY, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    free(buffer);

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);
    return ret;

}

static inline int is_aligned(const void *ptr, size_t page_size)
{
    return ((uintptr_t)ptr % page_size == 0);
}
#endif

static int cuda_put_hybrid(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data, double* itime)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    int ret = dspaces_SUCCESS;
    struct timeval start, end;
    double timer = 0; // timer in millisecond
    gettimeofday(&start, NULL);

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    // set the theshold = 48MB, > threshold using pipeline; < threshold using gdr
    size_t threshold = 48 << 20;

    cudaError_t curet;
    cudaStream_t stream;
    void* h_buffer;
    if(rdma_size >= threshold) {
        CUDA_ASSERTRT(cudaStreamCreate(&stream));
        h_buffer = (void*) malloc(rdma_size);
        curet = cudaMemcpyAsync(h_buffer, data, rdma_size, cudaMemcpyDeviceToHost, stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(h_buffer);
            return dspaces_ERR_CUDA;
        }
    }

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';
    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    in.odsc.size = sizeof(odsc);
    in.odsc.raw_odsc = (char *)(&odsc);
    in.odsc.gdim_size = sizeof(struct global_dimension);
    in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t hg_rdma_size = rdma_size;

    struct cudaPointerAttributes ptr_attr;
    struct hg_bulk_attr bulk_attr;

    if(rdma_size >= threshold) {
        hret = margo_bulk_create(client->mid, 1, (void **)&h_buffer, &hg_rdma_size,
                                    HG_BULK_READ_ONLY, &in.handle);
    } else {
        
    }

    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        cudaStreamDestroy(stream);
        free(h_buffer);
        return dspaces_ERR_MERCURY;
    }

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->put_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        cudaStreamDestroy(stream);
        free(h_buffer);
        return dspaces_ERR_MERCURY;
    }

    if(rdma_size >= threshold) {
        curet = cudaStreamSynchronize(stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(h_buffer);
            return dspaces_ERR_CUDA;
        }
    }

    gettimeofday(&end, NULL);
    timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
    *itime = timer;

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        cudaStreamDestroy(stream);
        free(h_buffer);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        cudaStreamDestroy(stream);
        free(h_buffer);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);

    if(rdma_size >= threshold) {
        CUDA_ASSERTRT(cudaStreamDestroy(stream));
        free(h_buffer);
    }

    return ret;
}

static int cuda_put_heuristic(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data, double* itime)
{
    /*  Choose to use conventional path or GDR path based on a score
        Performance score - 10 or 0
        The path that takes less time gains 10, the other path gains 0
        Heating score - 0 to 5(max)
        Artificially set the max to the half of the performance score
        If the path is not chosen, heating score +1
        Total score  = Performance Score + Heating Score
        Use Softmax of the total score for random choosing
    */
    double r;
    static int cnt = 0;
    cnt++;
    if(cnt < 3) { // 2 iters for warm-up
        srand((unsigned)time(NULL));
        r = ((double) rand() / (RAND_MAX));
        if(r < 0.5) { // choose host-based path
            return cuda_put_pipeline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        } else { // choose gdr path
            return cuda_put_gdr(client, var_name, ver, elem_size, ndim, lb, ub, data);
        }
    }
    int ret;
    struct timeval start, end;
    int perf_score = 10, heat_score_max = 5;

    struct bbox bb = {.num_dims = ndim};
    memset(bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memcpy(bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(bb.ub.c, ub, sizeof(uint64_t) * ndim);

    size_t rdma_size = elem_size * bbox_volume(&bb);
    

    struct gpu_bulk_list_entry *e;
    e = lookup_gpu_bulk_list(&client->dcg->gpu_bulk_put_list, rdma_size);
    if(!e) { // no record for this rdma size, randomly choose one of the path
        srand((unsigned)time(NULL));
        r = ((double) rand() / (RAND_MAX));
        e = (struct gpu_bulk_list_entry *) malloc(sizeof(*e));
        e->rdma_size = rdma_size;
        // each entry keeps 3 performance record
        for(int i=0; i<3; i++) {
            e->host_time[i] = -1.0;
            e->gdr_time[i] = -1.0;
        }
        e->host_heat_score = 0;
        e->gdr_heat_score = 0;
        if(r < 0.5) { // choose host-based path
            gettimeofday(&start, NULL);
            ret = cuda_put_pipeline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
            gettimeofday(&end, NULL);
            e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->host_heat_score = 0;
            e->gdr_heat_score =  e->gdr_heat_score < heat_score_max ? e->gdr_heat_score++ : heat_score_max;
        } else { // choose gdr path
            gettimeofday(&start, NULL);
            ret = cuda_put_gdr(client, var_name, ver, elem_size, ndim, lb, ub, data);
            gettimeofday(&end, NULL);
            e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->gdr_heat_score = 0;
            e->host_heat_score = e->host_heat_score < heat_score_max ? e->host_heat_score++ : heat_score_max;
        }
        list_add(&e->entry, &client->dcg->gpu_bulk_put_list);
    } else if(e->host_time[0] < 0) { // no record for host-based path, force to choose it
        gettimeofday(&start, NULL);
        ret = cuda_put_pipeline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        gettimeofday(&end, NULL);
        for(int i=2; i>0; i--) { // shift the record
            e->host_time[i] = e->host_time[i-1];
        }
        e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        e->host_heat_score = 0;
        e->gdr_heat_score =  e->gdr_heat_score < heat_score_max ? e->gdr_heat_score++ : heat_score_max;
    } else if(e->gdr_time[0] < 0) { // no record for gdr path, force to choose it
        gettimeofday(&start, NULL);
        ret = cuda_put_gdr(client, var_name, ver, elem_size, ndim, lb, ub, data);
        gettimeofday(&end, NULL);
        for(int i=2; i>0; i--) { // shift the record
            e->gdr_time[i] = e->gdr_time[i-1];
        }
        e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        e->gdr_heat_score = 0;
        e->host_heat_score = e->host_heat_score < heat_score_max ? e->host_heat_score++ : heat_score_max;
    } else { // have both records, choose the path according to score
        int host_perf_score, gdr_perf_score;
        double host_total_score, gdr_total_score, max_total_score;
        double avg_host_time = 0.0, avg_gdr_time = 0.0;
        int avg_host_cnt = 0, avg_gdr_cnt = 0;
        double host_prob, gdr_prob;
        for(int i=0; i<3; i++) {
            if(e->host_time[i] > 0.0) {
                avg_host_time += e->host_time[i];
                avg_host_cnt++;
            }
            if(e->gdr_time[i] > 0.0) {
                avg_gdr_time += e->gdr_time[i];
                avg_gdr_cnt++;
            }
        }
        avg_host_time /= avg_host_cnt;
        avg_gdr_time /= avg_gdr_cnt;
        if(avg_gdr_time > avg_host_time) { // host perf better
            host_perf_score = perf_score;
            gdr_perf_score = 0;
        } else { // gdr perf better
            host_perf_score = 0;
            gdr_perf_score = perf_score;
        }
        host_total_score = host_perf_score + e->host_heat_score;
        gdr_total_score = gdr_perf_score + e->gdr_heat_score;
        max_total_score = host_total_score > gdr_total_score ? host_total_score : gdr_total_score;
        host_prob = exp(host_total_score - max_total_score) / (exp(host_total_score -max_total_score)
                                                            + exp(gdr_total_score -max_total_score));
        gdr_prob = exp(gdr_total_score - max_total_score) / (exp(host_total_score -max_total_score)
                                                            + exp(gdr_total_score -max_total_score));
        DEBUG_OUT("host_prob = %lf, gdr_prob = %lf\n", host_prob, gdr_prob);
        srand((unsigned)time(NULL));
        double r = ((double) rand() / (RAND_MAX));
        if(r < host_prob) { // choose host-based path
            gettimeofday(&start, NULL);
            ret = cuda_put_pipeline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
            gettimeofday(&end, NULL);
            for(int i=2; i>0; i--) { // shift the record
                e->host_time[i] = e->host_time[i-1];
            }
            e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->host_heat_score = 0;
            e->gdr_heat_score =  e->gdr_heat_score < heat_score_max ? e->gdr_heat_score++ : heat_score_max;
        } else { // choose gdr path
            gettimeofday(&start, NULL);
            ret = cuda_put_gdr(client, var_name, ver, elem_size, ndim, lb, ub, data);
            gettimeofday(&end, NULL);
            for(int i=2; i>0; i--) { // shift the record
                e->gdr_time[i] = e->gdr_time[i-1];
            }
            e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->gdr_heat_score = 0;
            e->host_heat_score = e->host_heat_score < heat_score_max ? e->host_heat_score++ : heat_score_max;
        }
    }
    return ret;
}

static int cuda_put_dual_channel(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                const void *data, double* itime)
{
    /* 1 - Device -> Host cudaMemcpyAsync()
       2 - Device -> Remote Staging margo_iforward()
       3 - Wait Cuda Stream
       4 - Host -> Remote Staging margo_iforward()
       5 - Margo_wait_any() to measure the time for dual channel
       6 - Tune the ratio according to the time 
     */
    hg_addr_t server_addr;
    hg_handle_t gdr_handle, host_handle;
    hg_return_t hret;
    dc_bulk_gdim_t gdr_in, host_in;
    bulk_out_t gdr_out, host_out;
    int ret = dspaces_SUCCESS;
    struct timeval start, end;
    double timer = 0; // timer in millisecond
    gettimeofday(&start, NULL);

    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    size_t data_size = (elem_size)*bbox_volume(&odsc.bb);

    // preset data volume for gdr / pipeline = 50% : 50%
    // cut the data byte stream and record the offset
    static double gdr_ratio = 0.5;
    static double host_ratio = 0.5;

    size_t offset = (size_t) (gdr_ratio * data_size);
    size_t gdr_rdma_size = offset;
    size_t host_rdma_size = data_size - gdr_rdma_size;

    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreate(&stream));
    
    void * h_buffer = (void*) malloc(host_rdma_size);
    
    cudaError_t curet;
    curet = cudaMemcpyAsync(h_buffer, data+offset, host_rdma_size, cudaMemcpyDeviceToHost, stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        free(h_buffer);
        cudaStreamDestroy(stream);
        return dspaces_ERR_CUDA;
    }

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';
    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    gdr_in.odsc.size = sizeof(odsc);
    gdr_in.odsc.raw_odsc = (char *)(&odsc);
    gdr_in.odsc.gdim_size = sizeof(struct global_dimension);
    gdr_in.odsc.raw_gdim = (char *)(&odsc_gdim);
    gdr_in.channel = 0; /* gdr - 0 */
    gdr_in.offset = 0;
    gdr_in.rdma_size = gdr_rdma_size;

    get_server_address(client, &server_addr);
    margo_request gdr_req, host_req;

    hg_size_t hg_gdr_rdma_size = gdr_rdma_size;
    hg_size_t hg_host_rdma_size = host_rdma_size;

    struct cudaPointerAttributes ptr_attr;
    CUDA_ASSERTRT(cudaPointerGetAttributes(&ptr_attr, data));
    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                        .device = ptr_attr.device };

    hret = margo_bulk_create_attr(client->mid, 1, (void **)&data, &hg_gdr_rdma_size,
                                HG_BULK_READ_ONLY, &bulk_attr, &gdr_in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create_attr() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_create(client->mid, server_addr, client->put_dc_id, &gdr_handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_iforward(gdr_handle, &gdr_in, &gdr_req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_iforward() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        return dspaces_ERR_MERCURY;
    }

    host_in.odsc.size = sizeof(odsc);
    host_in.odsc.raw_odsc = (char *)(&odsc);
    host_in.odsc.gdim_size = sizeof(struct global_dimension);
    host_in.odsc.raw_gdim = (char *)(&odsc_gdim);
    host_in.channel = 1; /* host - 1 */
    host_in.offset = offset;
    host_in.rdma_size = host_rdma_size;

    hret = margo_bulk_create(client->mid, 1, (void **)&h_buffer, &hg_host_rdma_size,
                                    HG_BULK_READ_ONLY, &host_in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_create(client->mid, server_addr, client->put_dc_id, &host_handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        return dspaces_ERR_MERCURY;
    }

    curet = cudaStreamSynchronize(stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_CUDA;
    }

    gettimeofday(&end, NULL);
    timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
    *itime = timer;

    hret = margo_iforward(host_handle, &host_in, &host_req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_iforward() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    // struct timeval start, end;
    double gdr_timer, host_timer, wait_timer = 0; // timer in ms
    double epsilon = 1e-3; // 1us
    double lr = 1.0;

    /*  Try to tune the ratio every 2 timesteps
        At timestep (t), if 2nd timer(t) < 2e-6 s, means 2nd request(t) finishes no later than the 1st(t).
            Keep the same ratio at (t+1), but swap the request.
            If the 2nd timer(t+1) < 2e-6 s, means almost same time; else, tune the ratio and not swap request
            Suppose gdr finishes first initially: wait_flag = 0 -> gdr first; wait_flag = 1 -> host first
        else
    */
    margo_request *req0, *req1;
    double *timer0, *timer1;
    static int wait_flag = 0;
    if(wait_flag == 0) {
        req0 = &gdr_req;
        timer0 = &gdr_timer;
        req1 = &host_req;
        timer1 = &host_timer;
    } else {
        req0 = &host_req;
        timer0 = &host_timer;
        req1 = &gdr_req;
        timer1 = &gdr_timer;
    }

    gettimeofday(&start, NULL);
    hret = margo_wait(*req0);
    gettimeofday(&end, NULL);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_wait(): %s failed! Err Code: %d\n", __func__,
                            wait_flag == 0 ? "gdr_req":"host_req", hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }
    *timer0 = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;

    gettimeofday(&start, NULL);
    hret = margo_wait(*req1);
    gettimeofday(&end, NULL);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_wait(): %s failed! Err Code: %d\n", __func__,
                            wait_flag == 0 ? "host_req":"gdr_req", hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }
    *timer1 = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;

    if(*timer1 > 2e-3) {
        // 2nd request takes longer time, tune ratio
        if(gdr_timer < host_timer) {
            if(host_timer - gdr_timer > epsilon) {
                gdr_ratio += ((host_timer - gdr_timer) / host_timer) * lr * (1-gdr_ratio);
                host_ratio = 1 - gdr_ratio;
            }
        } else {
            if(gdr_timer - host_timer > epsilon) {
                gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * lr * (gdr_ratio-0);
                host_ratio = 1 - gdr_ratio;
            }
        }
    } else {
        // 2nd request finishes no later than the 1st request
        // swap request by setting flag = 1
        wait_flag == 0 ? 1:0;
    }

    DEBUG_OUT("ts = %u, gdr_ratio = %lf, host_ratio = %lf,"
                "gdr_time = %lf, host_time = %lf\n", ver, gdr_ratio, host_ratio, 
                    gdr_timer, host_timer);

    hret = margo_get_output(gdr_handle, &gdr_out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(host_handle, &host_out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    if(gdr_out.ret == 0 && host_out.ret == 0) {
        ret = dspaces_SUCCESS;
    } else {
        ret = dspaces_ERR_PUT;
    }

    margo_free_output(gdr_handle, &gdr_out);
    margo_free_output(host_handle, &host_out);

    margo_bulk_free(gdr_in.handle);
    margo_destroy(gdr_handle);
    margo_bulk_free(host_in.handle);
    margo_destroy(host_handle);
    
    margo_addr_free(client->mid, server_addr);

    return ret;
}

static int cuda_put_dual_channel_v2(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                void *data, double* itime)
{
    /* 1 - Device -> Host cudaMemcpyAsync()
       2 - Device -> Remote Staging margo_iforward()
       3 - Wait Cuda Stream
       4 - Host -> Remote Staging margo_iforward()
       5 - Margo_wait_any() to measure the time for dual channel
       6 - Tune the ratio according to the time 
    */

    // preset data volume for gdr / pipeline = 50% : 50%
    // cut the data along the highest dimension
    // first piece goes to host, second piece goes to GDR
    static double host_ratio = 0.5;
    static double gdr_ratio = 0.5;
    // 1MB makes no difference for single or dual channel
    uint64_t ratio_eps = 1 << 20;
    // make it to pure GDR or host-based when either the ratio is around 1 
    double min_ratio = host_ratio > gdr_ratio ? gdr_ratio : host_ratio;
    uint64_t put_rdma_size = elem_size;
    for(int i=0; i<ndim; i++) {
        put_rdma_size *= (ub[i] - lb[i] + 1);
    }
    if(min_ratio * put_rdma_size < ratio_eps) { // go to either pure GDR or host-based
        if(host_ratio > gdr_ratio) { // pure host
            return cuda_put_pipeline(client, var_name, ver, 
                            elem_size, ndim, lb, ub, data, itime);
        } else { // pure GDR
            *itime = 0;
            return cuda_put_gdr(client, var_name, ver, 
                            elem_size, ndim, lb, ub, data);
        }
    }

    hg_addr_t server_addr;
    hg_handle_t gdr_handle, host_handle;
    hg_return_t hret;
    bulk_gdim_t gdr_in, host_in;
    bulk_out_t gdr_out, host_out;
    margo_request gdr_req, host_req;
    int ret = dspaces_SUCCESS;
    struct timeval start, end;

    struct bbox host_bb = {.num_dims = ndim};
    memset(host_bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(host_bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memcpy(host_bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(host_bb.ub.c, ub, sizeof(uint64_t) * ndim);

    int cut_dim; // find the highest dimension whose dim length > 1
    for(int i=0; i<ndim; i++) {
        if(ub[i]-lb[i]>0) {
            cut_dim = i;
            break;
        }
    }

    uint64_t dist = ub[cut_dim] - lb[cut_dim] + 1;
    uint64_t cut_dist = dist * host_ratio;
    if(cut_dist == 0) { // host_ratio near zero, go to pure GDR
        *itime = 0;
        return cuda_put_gdr(client, var_name, ver, 
                        elem_size, ndim, lb, ub, data);
    } else if(cut_dist == dist) { // host_ratio near one, go to pure host-based
        return cuda_put_pipeline(client, var_name, ver, 
                            elem_size, ndim, lb, ub, data, itime);
    }
    host_bb.ub.c[cut_dim] = (uint64_t)(host_bb.lb.c[cut_dim] + cut_dist - 1);

    size_t host_rdma_size = (size_t) (elem_size*bbox_volume(&host_bb));

    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreate(&stream));
    
    void* h_buffer = (void*) malloc(host_rdma_size);
    
    /* Start D->H I/O ASAP */
    cudaError_t curet;
    curet = cudaMemcpyAsync(h_buffer, data, host_rdma_size, cudaMemcpyDeviceToHost, stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                __func__, cudaGetErrorString(curet));
        free(h_buffer);
        cudaStreamDestroy(stream);
        return dspaces_ERR_CUDA;
    }

    /* Start GDR I/O */
    obj_descriptor gdr_odsc = {.version = ver,
                               .owner = {0},
                               .st = st,
                               .flags = 0,
                               .size = elem_size};
    memcpy(&gdr_odsc.bb, &host_bb, sizeof(struct bbox));
    gdr_odsc.bb.lb.c[cut_dim] = host_bb.ub.c[cut_dim] + 1;
    memcpy(gdr_odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);
    strncpy(gdr_odsc.name, var_name, sizeof(gdr_odsc.name) - 1);
    gdr_odsc.name[sizeof(gdr_odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    gdr_in.odsc.size = sizeof(obj_descriptor);
    gdr_in.odsc.raw_odsc = (char *)(&gdr_odsc);
    gdr_in.odsc.gdim_size = sizeof(struct global_dimension);
    gdr_in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t hg_gdr_rdma_size = elem_size*bbox_volume(&gdr_odsc.bb);

    get_server_address(client, &server_addr);

    struct cudaPointerAttributes ptr_attr;
    CUDA_ASSERTRT(cudaPointerGetAttributes(&ptr_attr, data));
    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                        .device = ptr_attr.device };

    void* gdr_data = data+host_rdma_size;
    hret = margo_bulk_create_attr(client->mid, 1, (void **)&gdr_data, &hg_gdr_rdma_size,
                                HG_BULK_READ_ONLY, &bulk_attr, &gdr_in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create_attr() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_create(client->mid, server_addr, client->put_id, &gdr_handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_iforward(gdr_handle, &gdr_in, &gdr_req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_iforward() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        return dspaces_ERR_MERCURY;
    }

    /* Prep Host->Remote Staging I/O */
    obj_descriptor host_odsc = {.version = ver,
                               .owner = {0},
                               .st = st,
                               .flags = 0,
                               .size = elem_size};
    memcpy(&host_odsc.bb, &host_bb, sizeof(struct bbox));
    strncpy(host_odsc.name, var_name, sizeof(host_odsc.name) - 1);
    host_odsc.name[sizeof(host_odsc.name) - 1] = '\0';


    host_in.odsc.size = sizeof(obj_descriptor);
    host_in.odsc.raw_odsc = (char *)(&host_odsc);
    host_in.odsc.gdim_size = sizeof(struct global_dimension);
    host_in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t hg_host_rdma_size = host_rdma_size;

    hret = margo_bulk_create(client->mid, 1, (void **)&h_buffer, &hg_host_rdma_size,
                                    HG_BULK_READ_ONLY, &host_in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_create(client->mid, server_addr, client->put_id, &host_handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        return dspaces_ERR_MERCURY;
    }

    /* Sync Host Buffer */
    curet = cudaStreamSynchronize(stream);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_CUDA;
    }

    hret = margo_iforward(host_handle, &host_in, &host_req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_iforward() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    // struct timeval start, end;
    double gdr_timer, host_timer, wait_timer = 0; // timer in ms
    double epsilon = 1e-3; // 1us
    double lr = 1.0;

    /*  Try to tune the ratio every 2 timesteps
        At timestep (t), if 2nd timer(t) < 2e-6 s, means 2nd request(t) finishes no later than the 1st(t).
            Keep the same ratio at (t+1), but swap the request.
            If the 2nd timer(t+1) < 2e-6 s, means almost same time; else, tune the ratio and not swap request
            Suppose gdr finishes first initially: wait_flag = 0 -> gdr first; wait_flag = 1 -> host first
        else
    */
    margo_request *req0, *req1;
    double *timer0, *timer1;
    static int wait_flag = 0;
    if(wait_flag == 0) {
        req0 = &gdr_req;
        timer0 = &gdr_timer;
        req1 = &host_req;
        timer1 = &host_timer;
    } else {
        req0 = &host_req;
        timer0 = &host_timer;
        req1 = &gdr_req;
        timer1 = &gdr_timer;
    }

    gettimeofday(&start, NULL);
    hret = margo_wait(*req0);
    gettimeofday(&end, NULL);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_wait(): %s failed! Err Code: %d\n", __func__,
                            wait_flag == 0 ? "gdr_req":"host_req", hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }
    *timer0 = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;

    gettimeofday(&start, NULL);
    hret = margo_wait(*req1);
    gettimeofday(&end, NULL);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_wait(): %s failed! Err Code: %d\n", __func__,
                            wait_flag == 0 ? "host_req":"gdr_req", hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }
    *timer1 = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;

    if(*timer1 > 2e-3) {
        // 2nd request takes longer time, tune ratio
        if(gdr_timer < host_timer) {
            if(host_timer - gdr_timer > epsilon) {
                gdr_ratio += ((host_timer - gdr_timer) / host_timer) * lr * (1-gdr_ratio);
                host_ratio = 1 - gdr_ratio;
            }
        } else {
            if(gdr_timer - host_timer > epsilon) {
                gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * lr * (gdr_ratio-0);
                host_ratio = 1 - gdr_ratio;
            }
        }
    } else {
        // 2nd request finishes no later than the 1st request
        // swap request by setting flag = 1
        wait_flag == 0 ? 1:0;
    }

    DEBUG_OUT("ts = %u, gdr_ratio = %lf, host_ratio = %lf,"
                "gdr_time = %lf, host_time = %lf\n", ver, gdr_ratio, host_ratio, 
                    gdr_timer, host_timer);


    hret = margo_get_output(gdr_handle, &gdr_out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(host_handle, &host_out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed! Err Code: %d\n", __func__, hret);
        free(h_buffer);
        cudaStreamDestroy(stream);
        margo_bulk_free(gdr_in.handle);
        margo_destroy(gdr_handle);
        margo_bulk_free(host_in.handle);
        margo_destroy(host_handle);
        return dspaces_ERR_MERCURY;
    }

    if(gdr_out.ret == 0 && host_out.ret == 0) {
        ret = dspaces_SUCCESS;
    } else {
        ret = dspaces_ERR_PUT;
    }

    margo_free_output(gdr_handle, &gdr_out);
    margo_free_output(host_handle, &host_out);

    margo_bulk_free(gdr_in.handle);
    margo_destroy(gdr_handle);
    margo_bulk_free(host_in.handle);
    margo_destroy(host_handle);
    
    margo_addr_free(client->mid, server_addr);

    *itime = 0;
    return ret;
}

static int finalize_req(struct dspaces_put_req *req)
{
    bulk_out_t out;
    int ret;
    hg_return_t hret;

    hret = margo_get_output(req->handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed\n", __func__);
        margo_bulk_free(req->in.handle);
        margo_destroy(req->handle);
        return dspaces_ERR_MERCURY;
    }
    ret = out.ret;
    margo_free_output(req->handle, &out);
    margo_bulk_free(req->in.handle);
    margo_destroy(req->handle);

    if(req->buffer) {
        free(req->buffer);
        req->buffer = NULL;
    }

    req->finalized = 1;
    req->ret = ret;

    return ret;
}

static int cuda_put_dcds(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                void *data, double* itime)
{
    /*  cuda_put_dual_channel_dual_staging
        If the host has enough memory, offload the I/O to the host.
            Do dual channel for gdr and device->host
        If the host doesn't have enough memory, try to free the completed iput request;
        If the host still doesn't have enough memory, do gdr.
    */

    int ret = dspaces_SUCCESS;
    struct timeval start, end;

    struct bbox bb = {.num_dims = ndim};
    memset(bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memcpy(bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(bb.ub.c, ub, sizeof(uint64_t) * ndim);

    size_t rdma_size = elem_size * bbox_volume(&bb);

    void* host_buf = (void*) malloc(rdma_size);
    if(!host_buf) { // insufficient memory
        /* Since we merge put_local & iput & subscribe in 1 RPC call,
           There is no margo request for client to check if bulk transfer
           is finished. Client doesn't have to actively free host memory.
           Client can just wait the notification from server to free the
           host memory. So insufficient memory will directly go dual channel. */
        ret = cuda_put_dual_channel_v2(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
    } else { // GPU->host + put_local_sub_drain
        /* CUDA Async MemCpy */
        
        cudaStream_t stream;
        CUDA_ASSERTRT(cudaStreamCreate(&stream));
        cudaError_t curet;
        curet = cudaMemcpyAsync(host_buf, data, rdma_size, cudaMemcpyDeviceToHost, stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_CUDA;
        }

        hg_handle_t *handle = (hg_handle_t*) malloc(sizeof(hg_handle_t));
        hg_return_t hret;
        hg_addr_t server_addr;
        
        /* putlocal_subdrain Prep. */
        if(client->listener_init == 0) {
            ret = dspaces_init_listener(client);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): dspaces_init_listener() failed, "
                                "Err Code: (%d)\n",
                                __func__, ret);
                cudaStreamDestroy(stream);
                free(host_buf);
                free(handle);
                return (ret);
            }
        }

        obj_descriptor odsc = {.version = ver,
                                .st = st,
                                .flags = DS_CLIENT_STORAGE,
                                .size = elem_size};
        hg_addr_t owner_addr;
        size_t owner_addr_size = 128;
        margo_addr_self(client->mid, &owner_addr);
        margo_addr_to_string(client->mid, odsc.owner, &owner_addr_size, owner_addr);
        margo_addr_free(client->mid, owner_addr);
        memcpy(&odsc.bb, &bb, sizeof(struct bbox));
        strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
        odsc.name[sizeof(odsc.name) - 1] = '\0';

        struct obj_data *od;
        // allocate local od with host_buf
        od = obj_data_alloc_no_data(&odsc, host_buf);
        set_global_dimension(&(client->dcg->gdim_list), var_name,
                            &(client->dcg->default_gdim), &od->gdim);
        
        ABT_mutex_lock(client->ls_mutex);
        ls_add_obj(client->dcg->ls, od);
        DEBUG_OUT("Added into local_storage\n");
        client->local_put_count++;
        ABT_mutex_unlock(client->ls_mutex);

        bulk_gdim_t *in = (bulk_gdim_t*) malloc(sizeof(bulk_gdim_t));
        bulk_out_t out;

        in->odsc.size = sizeof(odsc);
        in->odsc.raw_odsc = (char *)(&odsc);
        in->odsc.gdim_size = sizeof(struct global_dimension);
        in->odsc.raw_gdim = (char *)(&od->gdim);
        hg_size_t hg_rdma_size = rdma_size;

        hret = margo_bulk_create(client->mid, 1, (void **)&host_buf, &hg_rdma_size,
                                HG_BULK_READ_ONLY, &(in->handle));
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
            cudaStreamDestroy(stream);
            free(host_buf);
            od->data = NULL;
            free(handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            return dspaces_ERR_MERCURY;
        }

        get_server_address(client, &server_addr);

        // create handle
        hret = margo_create(client->mid, server_addr, client->putlocal_subdrain_id, handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
            cudaStreamDestroy(stream);
            free(host_buf);
            od->data = NULL;
            free(handle);
            margo_addr_free(client->mid, server_addr);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            margo_bulk_free(in->handle);
            return dspaces_ERR_MERCURY;
        }

        gettimeofday(&start, NULL);
        // add req to putlocal_subdrain list
        struct subdrain_list_entry *e = (struct subdrain_list_entry*) malloc(sizeof(*e));
        e->odsc = odsc;
        e->buffer = host_buf;
        e->get_count = 0;
        e->bulk_handle = in;
        e->rpc_handle = handle;
        ABT_cond_create(&e->delete_cond);
        ABT_mutex_lock(client->putlocal_subdrain_mutex);
        list_add(&e->entry, &client->dcg->putlocal_subdrain_list);
        ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        /* putlocal_subdrain Prep. end*/

        /* Sync Device->Host I/O */
        curet = cudaStreamSynchronize(stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(host_buf);
            od->data = NULL;
            margo_addr_free(client->mid, server_addr);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            margo_bulk_free(in->handle);
            margo_destroy(*handle);
            free(handle);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            return dspaces_ERR_CUDA;
        }

        /* putlocal_subdrain RPC */
        hret = margo_forward(*handle, in);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
            cudaStreamDestroy(stream);
            free(host_buf);
            od->data = NULL;
            margo_addr_free(client->mid, server_addr);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            margo_bulk_free(in->handle);
            margo_destroy(*handle);
            free(handle);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            return dspaces_ERR_MERCURY;
        }

        hret = margo_get_output(*handle, &out);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s):  margo_get_output() failed! Err Code: %d\n", __func__, hret);
            cudaStreamDestroy(stream);
            free(host_buf);
            od->data = NULL;
            margo_addr_free(client->mid, server_addr);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            margo_bulk_free(in->handle);
            margo_destroy(*handle);
            free(handle);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            return dspaces_ERR_MERCURY;
        }

        if(out.ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: putlocal_subdrain_rpc() failed at the server\n");
            free(host_buf);
            od->data = NULL;
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            obj_data_free(od);
            margo_bulk_free(in->handle);
            margo_destroy(*handle);
            free(handle);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            ret = out.ret;
        }
        /* putlocal_subdrain RPC end */

        CUDA_ASSERTRT(cudaStreamDestroy(stream));
        margo_free_output(*handle, &out);
        margo_addr_free(client->mid, server_addr);
        *itime = 0;
    }
    return ret;
}

static int cuda_put_dcds_v2(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                void *data, double* itime)
{
    // preset data volume for host_based / GDR = 50% : 50%
    // cut the data along the highest dimension
    // first piece goes to host, second piece goes to GDR
    static double local_ratio = 0.5;
    static double gdr_ratio = 0.5;
    // 1MB makes no difference for single or dual channel
    uint64_t ratio_eps = 1 << 20;
    // make it to pure GDR or local_put when either the ratio is around 1 
    double min_ratio = local_ratio > gdr_ratio ? gdr_ratio : local_ratio;
    uint64_t put_rdma_size = elem_size;
    for(int i=0; i<ndim; i++) {
        put_rdma_size *= (ub[i] - lb[i] +1 );
    }
    if(min_ratio * put_rdma_size < ratio_eps) {
        if(local_ratio > gdr_ratio) { //pure dual staging
            return cuda_put_dcds(client, var_name, ver, elem_size,
                                    ndim, lb, ub, data, itime);
        } else { //pure GDR
            *itime = 0;
            return cuda_put_gdr(client, var_name, ver, 
                            elem_size, ndim, lb, ub, data);
        }
    }

    int ret = dspaces_SUCCESS;
    cudaError_t curet;
    hg_return_t hret;
    struct timeval start, end;

    cudaStream_t stream;
    struct cudaPointerAttributes ptr_attr;
    obj_descriptor local_odsc, gdr_odsc;
    struct global_dimension odsc_gdim;
    hg_addr_t server_addr, owner_addr;
    size_t owner_addr_size = 128;
    hg_handle_t gdr_handle, *host_handle;
    bulk_gdim_t gdr_in, *host_in;
    bulk_out_t gdr_out, host_out;
    margo_request gdr_req;
    hg_size_t hg_gdr_rdma_size, hg_host_rdma_size;
    struct hg_bulk_attr bulk_attr;
    void *gdr_data;
    struct obj_data *local_od;
    static int wait_flag = 0;
    double gdr_timer, host_timer;
    double lr = 1.0;

    struct bbox host_bb = {.num_dims = ndim};
    memset(host_bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(host_bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memcpy(host_bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(host_bb.ub.c, ub, sizeof(uint64_t) * ndim);

    int cut_dim; // find the highest dimension whose dim length > 1
    for(int i=0; i<ndim; i++) {
        if(ub[i]-lb[i]>0) {
            cut_dim = i;
            break;
        }
    }

    uint64_t dist = ub[cut_dim] - lb[cut_dim] + 1;
    uint64_t cut_dist = dist * local_ratio;
    if(cut_dist == 0) { // host_ratio near zero, go to pure GDR
        *itime = 0;
        return cuda_put_gdr(client, var_name, ver, 
                        elem_size, ndim, lb, ub, data);
    } else if(cut_dist == dist) { // host_ratio near one, go to pure host-based
        return cuda_put_pipeline(client, var_name, ver, 
                            elem_size, ndim, lb, ub, data, itime);
    }
    host_bb.ub.c[cut_dim] = (uint64_t)(host_bb.lb.c[cut_dim] + cut_dist - 1);

    size_t host_rdma_size = (size_t) (elem_size*bbox_volume(&host_bb));

    void* host_buf = (void*) malloc(host_rdma_size);
    if(!host_buf) { // insufficient memory
        /* Since we merge put_local & iput & subscribe in 1 RPC call,
           There is no margo request for client to check if bulk transfer
           is finished. Client doesn't have to actively free host memory.
           Client can just wait the notification from server to free the
           host memory. So insufficient memory will directly go dual channel. */
        ret = cuda_put_dual_channel_v2(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
    } else { // GPU->host + GDR + put_local_sub_drain
        /* Start D->H I/O ASAP */
        curet = cudaStreamCreate(&stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreate() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(host_buf);
            return dspaces_ERR_CUDA;
        }

        curet = cudaMemcpyAsync(host_buf, data, host_rdma_size, cudaMemcpyDeviceToHost, stream);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_CUDA;
        }

        /* Start GDR I/O */
        gdr_odsc.version = ver;
        memset(gdr_odsc.owner, 0, sizeof(gdr_odsc.owner));
        gdr_odsc.st = st;
        gdr_odsc.flags = 0;
        gdr_odsc.size = elem_size;
        memcpy(&gdr_odsc.bb, &host_bb, sizeof(struct bbox));
        gdr_odsc.bb.lb.c[cut_dim] = host_bb.ub.c[cut_dim] + 1;
        memcpy(gdr_odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);
        strncpy(gdr_odsc.name, var_name, sizeof(gdr_odsc.name) - 1);
        gdr_odsc.name[sizeof(gdr_odsc.name) - 1] = '\0';
        set_global_dimension(&(client->dcg->gdim_list), var_name,
                        &(client->dcg->default_gdim), &odsc_gdim);
        gdr_in.odsc.size = sizeof(obj_descriptor);
        gdr_in.odsc.raw_odsc = (char *)(&gdr_odsc);
        gdr_in.odsc.gdim_size = sizeof(struct global_dimension);
        gdr_in.odsc.raw_gdim = (char *)(&odsc_gdim);
        hg_gdr_rdma_size = elem_size*bbox_volume(&gdr_odsc.bb);

        curet = cudaPointerGetAttributes(&ptr_attr, data);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_CUDA;
        }
        bulk_attr.mem_type = HG_MEM_TYPE_CUDA;
        bulk_attr.device = ptr_attr.device;

        gdr_data = data+host_rdma_size;
        hret = margo_bulk_create_attr(client->mid, 1, (void **)&gdr_data, &hg_gdr_rdma_size,
                                HG_BULK_READ_ONLY, &bulk_attr, &gdr_in.handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                    "ERROR: (%s): margo_bulk_create_attr() failed! Err Code: %d\n",
                    __func__, hret);
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_MERCURY;
        }

        get_server_address(client, &server_addr);

        hret = margo_create(client->mid, server_addr, client->put_id, &gdr_handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                    "ERROR: (%s): margo_create() failed! Err Code: %d\n",
                    __func__, hret);
            margo_addr_free(client->mid, server_addr);
            margo_bulk_free(gdr_in.handle);
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_MERCURY;
        }

        hret = margo_iforward(gdr_handle, &gdr_in, &gdr_req);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                    "ERROR: (%s): margo_iforward() failed! Err Code: %d\n",
                    __func__, hret);
            margo_destroy(gdr_handle);
            margo_addr_free(client->mid, server_addr);
            margo_bulk_free(gdr_in.handle);
            cudaStreamDestroy(stream);
            free(host_buf);
            return dspaces_ERR_MERCURY;
        }

        /* Prep PutLocal_SubDrain RPC */
        if(client->listener_init == 0) {
            ret = dspaces_init_listener(client);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): dspaces_init_listener() failed, "
                                "Err Code: (%d)\n",
                                __func__, ret);
                margo_destroy(gdr_handle);
                margo_addr_free(client->mid, server_addr);
                margo_bulk_free(gdr_in.handle);
                cudaStreamDestroy(stream);
                free(host_buf);
                return (ret);
            }
        }

        local_odsc.version = ver;
        memset(local_odsc.owner, 0, sizeof(local_odsc.owner));
        local_odsc.st = st;
        local_odsc.flags = DS_CLIENT_STORAGE;
        local_odsc.size = elem_size;
        margo_addr_self(client->mid, &owner_addr);
        margo_addr_to_string(client->mid, local_odsc.owner, &owner_addr_size, owner_addr);
        margo_addr_free(client->mid, owner_addr);
        memcpy(&local_odsc.bb, &host_bb, sizeof(struct bbox));
        strncpy(local_odsc.name, var_name, sizeof(local_odsc.name) - 1);
        local_odsc.name[sizeof(local_odsc.name) - 1] = '\0';

        // allocate local od with host_buf
        local_od = obj_data_alloc_no_data(&local_odsc, host_buf);
        memcpy(&local_od->gdim, &odsc_gdim, sizeof(struct global_dimension));
        
        ABT_mutex_lock(client->ls_mutex);
        ls_add_obj(client->dcg->ls, local_od);
        client->local_put_count++;
        DEBUG_OUT("Added into local_storage\n");
        ABT_mutex_unlock(client->ls_mutex);

        host_in = (bulk_gdim_t*) malloc(sizeof(bulk_gdim_t));
        host_in->odsc.size = sizeof(local_odsc);
        host_in->odsc.raw_odsc = (char *)(&local_odsc);
        host_in->odsc.gdim_size = sizeof(struct global_dimension);
        host_in->odsc.raw_gdim = (char *)(&local_od->gdim);
        hg_host_rdma_size = host_rdma_size; 

        hret = margo_bulk_create(client->mid, 1, (void **)&host_buf, &hg_host_rdma_size,
                                HG_BULK_READ_ONLY, &(host_in->handle));
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed! Err Code: %d\n", __func__, hret);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_destroy(gdr_handle);
            margo_addr_free(client->mid, server_addr);
            margo_bulk_free(gdr_in.handle);
            cudaStreamDestroy(stream);
            return dspaces_ERR_MERCURY;
        }

        host_handle = (hg_handle_t*) malloc(sizeof(hg_handle_t));

        hret = margo_create(client->mid, server_addr, client->putlocal_subdrain_id, host_handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_create() failed! Err Code: %d\n", __func__, hret);
            free(host_handle);
            margo_bulk_free(host_in->handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_destroy(gdr_handle);
            margo_addr_free(client->mid, server_addr);
            margo_bulk_free(gdr_in.handle);
            cudaStreamDestroy(stream);
            return dspaces_ERR_MERCURY;
        }

        // add req to putlocal_subdrain list
        struct subdrain_list_entry *e = (struct subdrain_list_entry*) malloc(sizeof(*e));
        e->odsc = local_odsc;
        e->buffer = host_buf;
        e->get_count = 0;
        e->bulk_handle = host_in;
        e->rpc_handle = host_handle;
        ABT_cond_create(&e->delete_cond);
        ABT_mutex_lock(client->putlocal_subdrain_mutex);
        list_add(&e->entry, &client->dcg->putlocal_subdrain_list);
        ABT_mutex_unlock(client->putlocal_subdrain_mutex);

        /*  Try to tune the ratio every 2 timesteps
            At timestep (t), if 2nd timer(t) < 2e-6 s, means 2nd request(t) finishes no later than the 1st(t).
                Keep the same ratio at (t+1), but swap the request.
                If the 2nd timer(t+1) < 2e-6 s, means almost same time; else, tune the ratio and not swap request
                Suppose gdr finishes first initially: wait_flag = 0 -> gdr first; wait_flag = 1 -> host first
            else
        */

        if(wait_flag == 0) { // wait gdr first
            gettimeofday(&start, NULL);
            hret = margo_wait(gdr_req);
            gettimeofday(&end, NULL);
            if(hret != HG_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): margo_wait() failed! Err Code: %d\n",
                            __func__, hret);
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_destroy(gdr_handle);
                margo_addr_free(client->mid, server_addr);
                margo_bulk_free(gdr_in.handle);
                cudaStreamDestroy(stream);
                return dspaces_ERR_MERCURY;
            }
            margo_bulk_free(gdr_in.handle);
            gdr_timer = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;

            /* Sync Device->Host I/O */
            gettimeofday(&start, NULL);
            curet = cudaStreamSynchronize(stream);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_addr_free(client->mid, server_addr);
                cudaStreamDestroy(stream);
                return dspaces_ERR_CUDA;
            }
            /* putlocal_subdrain RPC */
            hret = margo_forward(*host_handle, host_in);
            if(hret != HG_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_addr_free(client->mid, server_addr);
                cudaStreamDestroy(stream);
                return dspaces_ERR_MERCURY;
            }
            gettimeofday(&end, NULL);
            cudaStreamDestroy(stream);
            host_timer = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            fprintf(stdout, "ts = %u, gdr_ratio = %lf, local_ratio = %lf,"
                    "gdr_timer = %lf, host_timer = %lf\n", ver, gdr_ratio, local_ratio,
                    gdr_timer, host_timer);
            if(host_timer > 2) { // host timer > 2ms
                // host path takes longer time, tune ratio
                gdr_ratio += ((host_timer - gdr_timer) / host_timer) * lr * (1-gdr_ratio);
                local_ratio = 1 - gdr_ratio;
            } else {
                // host request finishes no later than the GDR request
                // swap request by setting flag = 1
                wait_flag = 1;
            }
        } else { // wait D->H first
            /* Sync Device->Host I/O */
            gettimeofday(&start, NULL);
            curet = cudaStreamSynchronize(stream);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_destroy(gdr_handle);
                margo_addr_free(client->mid, server_addr);
                margo_bulk_free(gdr_in.handle);
                cudaStreamDestroy(stream);
                return dspaces_ERR_CUDA;
            }
            /* putlocal_subdrain RPC */
            hret = margo_forward(*host_handle, host_in);
            if(hret != HG_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): margo_forward() failed! Err Code: %d\n", __func__, hret);
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_destroy(gdr_handle);
                margo_addr_free(client->mid, server_addr);
                margo_bulk_free(gdr_in.handle);
                cudaStreamDestroy(stream);
                return dspaces_ERR_MERCURY;
            }
            gettimeofday(&end, NULL);
            cudaStreamDestroy(stream);
            host_timer = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        
            gettimeofday(&start, NULL);
            hret = margo_wait(gdr_req);
            gettimeofday(&end, NULL);
            if(hret != HG_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): margo_wait() failed! Err Code: %d\n",
                            __func__, hret);
                ABT_cond_free(&e->delete_cond);
                ABT_mutex_lock(client->putlocal_subdrain_mutex);
                list_del(&e->entry);
                ABT_mutex_unlock(client->putlocal_subdrain_mutex);
                margo_destroy(*host_handle);
                free(host_handle);
                margo_bulk_free(host_in->handle);
                ABT_mutex_lock(client->ls_mutex);
                ls_remove(client->dcg->ls, local_od);
                client->local_put_count--;
                ABT_mutex_unlock(client->ls_mutex);
                free(host_buf);
                local_od->data = NULL;
                obj_data_free(local_od);
                margo_destroy(gdr_handle);
                margo_addr_free(client->mid, server_addr);
                margo_bulk_free(gdr_in.handle);
            }
            margo_bulk_free(gdr_in.handle);
            gdr_timer = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            fprintf(stdout, "ts = %u, gdr_ratio = %lf, local_ratio = %lf,"
                    "gdr_timer = %lf, host_timer = %lf\n", ver, gdr_ratio, local_ratio,
                    gdr_timer, host_timer);
            if(gdr_timer > 2) { // GDR timer > 2ms
                // GDR path takes longer time, tune ratio
                gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * lr * (gdr_ratio-0);
                local_ratio = 1 - gdr_ratio;
            } else {
                // GDR request finishes no later than the host request
                // swap request by setting flag = 0
                wait_flag = 0;
            }
        }

        hret = margo_get_output(gdr_handle, &gdr_out);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s):  margo_get_output() failed! Err Code: %d\n", __func__, hret);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            margo_destroy(*host_handle);
            fprintf(stderr, "DEBUG0\n");
            free(host_handle);
            margo_bulk_free(host_in->handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            fprintf(stderr, "DEBUG1\n");
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_addr_free(client->mid, server_addr);
            fprintf(stderr, "DEBUG2\n");
            return dspaces_ERR_MERCURY;
        }

        if(gdr_out.ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s):  put_rpc() failed at the server! Err Code: %d\n", __func__, gdr_out.ret);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            margo_destroy(*host_handle);
            free(host_handle);
            margo_bulk_free(host_in->handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_addr_free(client->mid, server_addr);
            return gdr_out.ret;
        }

        hret = margo_get_output(*host_handle, &host_out);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s):  margo_get_output() failed! Err Code: %d\n", __func__, hret);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            margo_destroy(*host_handle);
            fprintf(stderr, "DEBUG3\n");
            free(host_handle);
            margo_bulk_free(host_in->handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            fprintf(stderr, "DEBUG4\n");
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_addr_free(client->mid, server_addr);
            fprintf(stderr, "DEBUG5\n");
            return dspaces_ERR_MERCURY;
        }

        if(host_out.ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s):  putlocal_subdrain_rpc() failed at the server! Err Code: %d\n", __func__, host_out.ret);
            ABT_cond_free(&e->delete_cond);
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            list_del(&e->entry);
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
            margo_destroy(*host_handle);
            free(host_handle);
            margo_bulk_free(host_in->handle);
            ABT_mutex_lock(client->ls_mutex);
            ls_remove(client->dcg->ls, local_od);
            client->local_put_count--;
            ABT_mutex_unlock(client->ls_mutex);
            free(host_buf);
            local_od->data = NULL;
            obj_data_free(local_od);
            margo_addr_free(client->mid, server_addr);
            return host_out.ret;
        }

        margo_free_output(gdr_handle, &gdr_out);
        margo_free_output(*host_handle, &host_out);
        margo_destroy(gdr_handle);
        margo_addr_free(client->mid, server_addr);
        *itime = local_ratio;
    }
    return ret;
}

static void notify_drain_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    odsc_list_t in;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);
    
    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc_list.raw_odsc, sizeof(obj_descriptor));
    
    DEBUG_OUT("Received drain finished notification for obj %s\n",
                obj_desc_sprint(&in_odsc));

    ABT_mutex_lock(client->putlocal_subdrain_mutex);

    struct subdrain_list_entry *e =
        lookup_putlocal_subdrain_list(&client->dcg->putlocal_subdrain_list, in_odsc);
    margo_bulk_free(e->bulk_handle->handle);
    free(e->bulk_handle);
    margo_destroy(*(e->rpc_handle));
    while(e->get_count > 0) { // in case any pending get
        ABT_cond_wait(e->delete_cond, client->putlocal_subdrain_mutex);
    }

    ABT_mutex_lock(client->ls_mutex);
    struct obj_data *od = ls_find(client->dcg->ls, &e->odsc);
    if(od) {
        ls_remove(client->dcg->ls, od);
    } else {
        free(e->buffer);
        e->buffer = NULL;
    }
    ABT_mutex_unlock(client->ls_mutex);

    // remove entry
    ABT_cond_free(&e->delete_cond);
    list_del(&e->entry);
    free(e);
    client->local_put_count--;

    ABT_mutex_unlock(client->putlocal_subdrain_mutex);

    margo_free_input(handle, &in);
    margo_destroy(handle);

    if(client->local_put_count == 0 && client->f_final) {
        DEBUG_OUT("signaling all objects drained.\n");
        ABT_cond_signal(client->drain_cond);
    }
}
DEFINE_MARGO_RPC_HANDLER(notify_drain_rpc)

int dspaces_cuda_put(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                void *data, double* itime)
{
    int ret = dspaces_SUCCESS;

    switch (client->cuda_info.cuda_put_mode)
    {
    case 0:
        ret = cuda_put_hybrid(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 1:
        ret = cuda_put_baseline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 2:
        ret = cuda_put_pipeline(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 3:
        ret = cuda_put_gdr(client, var_name, ver, elem_size, ndim, lb, ub, data);
        break;
#ifdef HAVE_GDRCOPY
    case 4:
        // check if data pointer is aligned to GPU page defined in gdrcopy
        if(is_aligned(data, GPU_PAGE_SIZE)) {
            ret = cuda_put_gdrcopy(client, var_name, ver, elem_size, ndim, lb, ub, data);
        } else {
            ret = cuda_put_hybrid(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        }
        break;
#endif
    case 5:
        ret = cuda_put_heuristic(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 6:
        ret = cuda_put_dual_channel_v2(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 7:
        ret = cuda_put_dcds(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    case 8:
        ret = cuda_put_dcds_v2(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    default:
        ret = cuda_put_hybrid(client, var_name, ver, elem_size, ndim, lb, ub, data, itime);
        break;
    }

    return ret;
}

int dspaces_put(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub,
                void *data)
{
    int ret;
    double itime;
    struct cudaPointerAttributes ptr_attr;
    CUDA_ASSERTRT(cudaPointerGetAttributes(&ptr_attr, data));
    if(ptr_attr.type == cudaMemoryTypeDevice) {
        ret = dspaces_cuda_put(client, var_name, ver, elem_size, ndim, lb, ub, data, &itime);
    } else {
        ret = dspaces_cpu_put(client, var_name, ver, elem_size, ndim, lb, ub, data);
    }
    return ret;
}

struct dspaces_put_req *dspaces_iput(dspaces_client_t client,
                                     const char *var_name, unsigned int ver,
                                     int elem_size, int ndim, uint64_t *lb,
                                     uint64_t *ub, void *data, int alloc,
                                     int check, int free)
{
    hg_addr_t server_addr;
    hg_return_t hret;
    struct dspaces_put_req *ds_req, *ds_req_prev, **ds_req_p;
    int ret = dspaces_SUCCESS;
    const void *buffer;
    int flag;

    if(check) {
        // Check for comleted iputs
        ds_req_prev = NULL;
        ds_req_p = &client->put_reqs;
        while(*ds_req_p) {
            ds_req = *ds_req_p;
            flag = 0;
            if(!ds_req->finalized) {
                margo_test(ds_req->req, &flag);
                if(flag) {
                    finalize_req(ds_req);
                    if(!ds_req->next) {
                        client->put_reqs_end = ds_req_prev;
                    }
                    *ds_req_p = ds_req->next;
                    // do not free ds_req yet - user might do a
                    // dspaces_check_put later
                }
            }
            ds_req_prev = ds_req;
            ds_req_p = &ds_req->next;
        }
    }

    ds_req = calloc(1, sizeof(*ds_req));
    obj_descriptor odsc = {.version = ver,
                           .owner = {0},
                           .st = st,
                           .flags = 0,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    struct global_dimension odsc_gdim;
    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &odsc_gdim);

    ds_req->in.odsc.size = sizeof(odsc);
    ds_req->in.odsc.raw_odsc = (char *)(&odsc);
    ds_req->in.odsc.gdim_size = sizeof(struct global_dimension);
    ds_req->in.odsc.raw_gdim = (char *)(&odsc_gdim);
    hg_size_t rdma_size = (elem_size)*bbox_volume(&odsc.bb);

    if(alloc) {
        ds_req->buffer = malloc(rdma_size);
        memcpy(ds_req->buffer, data, rdma_size);
        buffer = ds_req->buffer;
    } else {
        buffer = data;
        if(free) {
            ds_req->buffer = data;
        }
    }

    DEBUG_OUT("sending object %s \n", obj_desc_sprint(&odsc));

    hret = margo_bulk_create(client->mid, 1, (void **)&buffer, &rdma_size,
                             HG_BULK_READ_ONLY, &ds_req->in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed\n", __func__);
        return dspaces_PUT_NULL;
    }

    get_server_address(client, &server_addr);

    hret =
        margo_create(client->mid, server_addr, client->put_id, &ds_req->handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        margo_bulk_free(ds_req->in.handle);
        return dspaces_PUT_NULL;
    }

    hret = margo_iforward(ds_req->handle, &ds_req->in, &ds_req->req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed\n", __func__);
        margo_bulk_free(ds_req->in.handle);
        margo_destroy(ds_req->handle);
        return dspaces_PUT_NULL;
    }

    margo_addr_free(client->mid, server_addr);

    ds_req->next = NULL;
    if(client->put_reqs_end) {
        client->put_reqs_end->next = ds_req;
        client->put_reqs_end = ds_req;
    } else {
        client->put_reqs = client->put_reqs_end = ds_req;
    }

    return ds_req;
}

int dspaces_check_put(dspaces_client_t client, struct dspaces_put_req *req,
                      int wait)
{
    int flag;
    struct dspaces_put_req **ds_req_p, *ds_req_prev;
    int ret;
    hg_return_t hret;

    if(req->finalized) {
        ret = req->ret;
        free(req);
        return ret;
    }

    if(wait) {
        hret = margo_wait(req->req);
        if(hret == HG_SUCCESS) {
            ds_req_prev = NULL;
            ds_req_p = &client->put_reqs;
            while(*ds_req_p && *ds_req_p != req) {
                ds_req_prev = *ds_req_p;
                ds_req_p = &((*ds_req_p)->next);
            }
            if(!ds_req_p) {
                fprintf(stderr,
                        "ERROR: put req finished, but was not saved.\n");
                return (-1);
            } else {
                ret = finalize_req(req);
                if(req->next == NULL) {
                    client->put_reqs_end = ds_req_prev;
                }
                *ds_req_p = req->next;
                free(req);
                return ret;
            }
        }
    } else {
        margo_test(req->req, &flag);
        if(flag) {
            ds_req_prev = NULL;
            ds_req_p = &client->put_reqs;
            while(*ds_req_p && *ds_req_p != req) {
                ds_req_prev = *ds_req_p;
                ds_req_p = &((*ds_req_p)->next);
            }
            if(!ds_req_p) {
                fprintf(stderr,
                        "ERROR: put req finished, but was not saved.\n");
                return (-1);
            } else {
                ret = finalize_req(req);
                if(req->next == NULL) {
                    client->put_reqs_end = ds_req_prev;
                }
                *ds_req_p = req->next;
                free(req);
            }
        }
        return flag;
    }
}

static void fill_odsc(const char *var_name, unsigned int ver, int elem_size,
                      int ndim, uint64_t *lb, uint64_t *ub,
                      obj_descriptor *odsc)
{
    odsc->version = ver;
    memset(odsc->owner, 0, sizeof(odsc->owner));
    odsc->st = st;
    odsc->size = elem_size;
    odsc->bb.num_dims = ndim;

    memset(odsc->bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc->bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc->bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc->bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc->name, var_name, sizeof(odsc->name) - 1);
    odsc->name[sizeof(odsc->name) - 1] = '\0';
}

static int get_data(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *data)
{
    bulk_in_t *in;
    in = (bulk_in_t *)malloc(sizeof(bulk_in_t) * num_odscs);

    struct obj_data **od;
    od = malloc(num_odscs * sizeof(struct obj_data *));

    margo_request *serv_req;
    hg_handle_t *hndl;
    hndl = (hg_handle_t *)malloc(sizeof(hg_handle_t) * num_odscs);
    serv_req = (margo_request *)malloc(sizeof(margo_request) * num_odscs);

    for(int i = 0; i < num_odscs; ++i) {
        od[i] = obj_data_alloc(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        hg_size_t rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);

        margo_bulk_create(client->mid, 1, (void **)(&(od[i]->data)), &rdma_size,
                          HG_BULK_WRITE_ONLY, &in[i].handle);

        hg_addr_t server_addr;
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);

        hg_handle_t handle;
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &handle);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &handle);
        }
        margo_request req;
        // forward get requests
        margo_iforward(handle, &in[i], &req);
        hndl[i] = handle;
        serv_req[i] = req;
        margo_addr_free(client->mid, server_addr);
    }

    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, data);

    // TODO: rewrite with margo_wait_any()
    for(int i = 0; i < num_odscs; ++i) {
        margo_wait(serv_req[i]);
        bulk_out_t resp;
        margo_get_output(hndl[i], &resp);
        margo_free_output(hndl[i], &resp);
        margo_destroy(hndl[i]);
        // copy received data into user return buffer
        ssd_copy(return_od, od[i]);
        obj_data_free(od[i]);
    }
    free(hndl);
    free(serv_req);
    free(in);
    free(return_od);

    return 0;
}

static int get_data_baseline(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *data, double *ctime)
{
    struct timeval start, end;
    double timer = 0; // timer in second
    bulk_in_t *in;
    in = (bulk_in_t *)malloc(sizeof(bulk_in_t) * num_odscs);

    struct obj_data **od;
    od = malloc(num_odscs * sizeof(struct obj_data *));

    margo_request *serv_req;
    hg_handle_t *hndl;
    hndl = (hg_handle_t *)malloc(sizeof(hg_handle_t) * num_odscs);
    serv_req = (margo_request *)malloc(sizeof(margo_request) * num_odscs);

    for(int i = 0; i < num_odscs; ++i) {
        od[i] = obj_data_alloc(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        hg_size_t rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);

        margo_bulk_create(client->mid, 1, (void **)(&(od[i]->data)), &rdma_size,
                          HG_BULK_WRITE_ONLY, &in[i].handle);

        hg_addr_t server_addr;
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);

        hg_handle_t handle;
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &handle);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &handle);
        }
        margo_request req;
        // forward get requests
        margo_iforward(handle, &in[i], &req);
        hndl[i] = handle;
        serv_req[i] = req;
        margo_addr_free(client->mid, server_addr);
    }

    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, data);

    // TODO: rewrite with margo_wait_any()
    for(int i = 0; i < num_odscs; ++i) {
        margo_wait(serv_req[i]);
        bulk_out_t resp;
        margo_get_output(hndl[i], &resp);
        margo_free_output(hndl[i], &resp);
        margo_bulk_free(in[i].handle);
        margo_destroy(hndl[i]);
        // copy received data into user return buffer
        gettimeofday(&start, NULL);
        ssd_copy(return_od, od[i]);
        gettimeofday(&end, NULL);
        timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        obj_data_free(od[i]);
    }
    free(hndl);
    free(serv_req);
    free(od);
    free(in);
    free(return_od);

    *ctime = timer;
    return 0;
}

static int get_data_gdr(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    struct timeval start, end;
    double timer = 0; // timer in second
    int ret = dspaces_SUCCESS;
    bulk_in_t *in;
    in = (bulk_in_t *)malloc(sizeof(bulk_in_t) * num_odscs);

    struct obj_data **od;
    od = malloc(num_odscs * sizeof(struct obj_data *));

    margo_request *serv_req;
    hg_handle_t *hndl;
    hndl = (hg_handle_t *)malloc(sizeof(hg_handle_t) * num_odscs);
    serv_req = (margo_request *)malloc(sizeof(margo_request) * num_odscs);

    struct hg_bulk_attr *bulk_attr;
    bulk_attr = (struct hg_bulk_attr*) malloc(sizeof(struct hg_bulk_attr) * num_odscs);

    cudaError_t curet;
    struct cudaPointerAttributes ptr_attr;
    curet = cudaPointerGetAttributes(&ptr_attr, d_data);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                __func__, cudaGetErrorString(curet));
        free(bulk_attr);
        free(serv_req);
        free(hndl);
        free(od);
        free(in);
        return dspaces_ERR_CUDA;
    }

    for(int i = 0; i < num_odscs; ++i) {
        od[i] = obj_data_alloc_cuda(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        hg_size_t rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);

        bulk_attr[i] = (struct hg_bulk_attr) {.mem_type = HG_MEM_TYPE_CUDA,
                                                .device = ptr_attr.device};
        margo_bulk_create_attr(client->mid, 1, (void **)(&(od[i]->data)),
                            &rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr[i],
                            &in[i].handle);

        hg_addr_t server_addr;
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);

        hg_handle_t handle;
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &handle);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &handle);
        }
        margo_request req;
        // forward get requests
        margo_iforward(handle, &in[i], &req);
        hndl[i] = handle;
        serv_req[i] = req;
        margo_addr_free(client->mid, server_addr);
    }

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    // concurrent cuda streams assigned to each od
    cudaStream_t* stream;
    int stream_size;
    if(client->cuda_info.concurrency_enabled) {
        if(num_odscs < client->cuda_info.num_concurrent_kernels) {
            stream_size = num_odscs;
        } else {
            stream_size = client->cuda_info.num_concurrent_kernels;
        }

        stream = (cudaStream_t*) malloc(stream_size*sizeof(cudaStream_t));
        for(int i = 0; i < stream_size; i++) {
            curet = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(bulk_attr);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(od[i]);
                }
                free(od);
                free(in);
                return dspaces_ERR_CUDA;
            }
        }
    }

    for(int i = 0; i < num_odscs; ++i) {
        margo_wait(serv_req[i]);
        bulk_out_t resp;
        margo_get_output(hndl[i], &resp);
        margo_free_output(hndl[i], &resp);
        margo_bulk_free(in[i].handle);
        margo_destroy(hndl[i]);
        // copy received data into user return buffer
        if(client->cuda_info.concurrency_enabled) {
            gettimeofday(&start, NULL);
            ret = ssd_copy_cuda_async(return_od, od[i], &stream[i%stream_size]);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                        __func__, ret);
                free(stream);
                free(return_od);
                free(bulk_attr);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(od[i]);
                }
                free(od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            gettimeofday(&end, NULL);
        } else {
            gettimeofday(&start, NULL);
            ret = ssd_copy_cuda(return_od, od[i]);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): ssd_copy_cuda() failed, Err Code: (%d)\n",
                        __func__, ret);
                free(stream);
                free(return_od);
                free(bulk_attr);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(od[i]);
                }
                free(od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            gettimeofday(&end, NULL);
            obj_data_free_cuda(od[i]);
        }
        timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;  
    }

    if(client->cuda_info.concurrency_enabled) {
        gettimeofday(&start, NULL);
        for(int i = 0; i < stream_size; i++) {
            curet = cudaStreamSynchronize(stream[i]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(od[i]);
                }
                free(od);
                free(in);
                return dspaces_ERR_CUDA;
            }

            curet = cudaStreamDestroy(stream[i]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamDestroy() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(od[i]);
                }
                free(od);
                free(in);
                return dspaces_ERR_CUDA;
            }
        }
        gettimeofday(&end, NULL);
        timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        free(stream);

        for(int i = 0; i < num_odscs; i++) {
            obj_data_free_cuda(od[i]);
        }
    }

    free(bulk_attr);
    free(hndl);
    free(serv_req);
    free(od);
    free(in);
    free(return_od);

    *ctime = timer;
    return ret;
}

static int get_data_hybrid(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    struct timeval start, end;
    double timer = 0; // timer in second
    int ret = dspaces_SUCCESS;
    bulk_in_t *in;
    in = (bulk_in_t *)malloc(sizeof(bulk_in_t) * num_odscs);

    struct obj_data **host_od;
    host_od = malloc(num_odscs * sizeof(struct obj_data *));
    
    margo_request *serv_req;
    hg_handle_t *hndl;
    hndl = (hg_handle_t *)malloc(sizeof(hg_handle_t) * num_odscs);
    serv_req = (margo_request *)malloc(sizeof(margo_request) * num_odscs);

    for(int i = 0; i < num_odscs; ++i) {
        host_od[i] = obj_data_alloc(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        hg_size_t rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);

        margo_bulk_create(client->mid, 1, (void **)(&(host_od[i]->data)), &rdma_size,
                          HG_BULK_WRITE_ONLY, &in[i].handle);

        hg_addr_t server_addr;
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);

        hg_handle_t handle;
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &handle);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &handle);
        }
        margo_request req;
        // forward get requests
        margo_iforward(handle, &in[i], &req);
        hndl[i] = handle;
        serv_req[i] = req;
        margo_addr_free(client->mid, server_addr);
    }

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    cudaError_t curet;
    // concurrent cuda streams assigned to each od
    cudaStream_t* stream;
    int stream_size;
    if(client->cuda_info.concurrency_enabled) {
        if(num_odscs < client->cuda_info.num_concurrent_kernels) {
            stream_size = num_odscs;
        } else {
            stream_size = client->cuda_info.num_concurrent_kernels;
        }

        stream = (cudaStream_t*) malloc(stream_size*sizeof(cudaStream_t));
        for(int i = 0; i < stream_size; i++) {
            curet = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
        }
    }

    struct obj_data **device_od;
    device_od = malloc(num_odscs * sizeof(struct obj_data *));

    // TODO: rewrite with margo_wait_any()
    for(int i = 0; i < num_odscs; ++i) {
        device_od[i] = obj_data_alloc_cuda(&odsc_tab[i]);
        margo_wait(serv_req[i]);
        bulk_out_t resp;
        margo_get_output(hndl[i], &resp);
        // H->D async transfer
        size_t data_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);
        if(client->cuda_info.concurrency_enabled) {
            curet = cudaMemcpyAsync(device_od[i]->data, host_od[i]->data, data_size,
                                    cudaMemcpyHostToDevice, stream[i%stream_size]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(device_od);
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            gettimeofday(&start, NULL);
            ret =  ssd_copy_cuda_async(return_od, device_od[i], &stream[i%stream_size]);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                        __func__, ret);
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            gettimeofday(&end, NULL);
        } else {
            curet = cudaMemcpy(device_od[i]->data, host_od[i]->data, data_size,
                                cudaMemcpyHostToDevice);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaMemcpy() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            obj_data_free(host_od[i]);
            gettimeofday(&start, NULL);
            ret = ssd_copy_cuda(return_od, device_od[i]);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): ssd_copy_cuda() failed, Err Code: (%d)\n",
                        __func__, ret);
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
            gettimeofday(&end, NULL);
            obj_data_free_cuda(device_od[i]);
        }

        timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        margo_free_output(hndl[i], &resp);
        margo_bulk_free(in[i].handle);
        margo_destroy(hndl[i]);
    }

    if(client->cuda_info.concurrency_enabled) {
        for(int i = 0; i < stream_size; i++) {
            curet = cudaStreamSynchronize(stream[i]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }

            curet = cudaStreamDestroy(stream[i]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamDestroy() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                free(stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int i = 0; i < num_odscs; i++) {
                    obj_data_free_cuda(device_od[i]);
                    obj_data_free(host_od[i]);
                }
                free(host_od);
                free(in);
                return dspaces_ERR_CUDA;
            }
        }

        free(stream);

        for(int i = 0; i < num_odscs; i++) {
            obj_data_free_cuda(device_od[i]);
            obj_data_free(host_od[i]);
        }
    }

    free(device_od);
    free(hndl);
    free(serv_req);
    free(host_od);
    free(in);
    free(return_od);

    *ctime = timer;
    return ret;
}

static int get_data_heuristic(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    /*  Choose to use conventional path or GDR path based on a score
        Performance score - 10 or 0
        The path that takes less time gains 10, the other path gains 0
        Heating score - 0 to 5(max)
        Artificially set the max to the half of the performance score
        If the path is not chosen, heating score +1
        Total score  = Performance Score + Heating Score
        Use Softmax of the total score for random choosing
    */
    double r;
    static int cnt = 0;
    cnt++;
    if(cnt < 3) { // 2 iters for warm-up
        srand((unsigned)time(NULL));
        r = ((double) rand() / (RAND_MAX));
        if(r < 0.5) { // choose host-based path
            return get_data_hybrid(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
        } else { // choose gdr path
            return get_data_gdr(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
        }
    }
    int ret = dspaces_SUCCESS;
    struct timeval start, end;

    size_t rdma_size = req_obj.size*bbox_volume(&req_obj.bb);
    struct gpu_bulk_list_entry *e;
    e = lookup_gpu_bulk_list(&client->dcg->gpu_bulk_get_list, rdma_size);
    if(!e) { // no record for this rdma size, randomly choose one of the path
        srand((unsigned)time(NULL));
        double r = ((double) rand() / (RAND_MAX));
        e = (struct gpu_bulk_list_entry *) malloc(sizeof(*e));
        e->rdma_size = rdma_size;
        // each entry keeps 3 performance record
        for(int i=0; i<3; i++) {
            e->host_time[i] = -1.0;
            e->gdr_time[i] = -1.0;
        }
        e->host_heat_score = 0;
        e->gdr_heat_score = 0;
        if(r < 0.5) { // choose host-based path
            gettimeofday(&start, NULL);
            ret = get_data_hybrid(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
            gettimeofday(&end, NULL);
            e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->host_heat_score = 0;
            e->gdr_heat_score =  e->gdr_heat_score < 5 ? e->gdr_heat_score++ : 5;
        } else { // choose gdr path
            gettimeofday(&start, NULL);
            ret = get_data_gdr(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
            gettimeofday(&end, NULL);
            e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->gdr_heat_score = 0;
            e->host_heat_score = e->host_heat_score < 5 ? e->host_heat_score++ : 5;
        }
        list_add(&e->entry, &client->dcg->gpu_bulk_get_list);
    } else if(e->host_time[0] < 0) { // no record for host-based path, force to choose it
        gettimeofday(&start, NULL);
        ret = get_data_hybrid(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
        gettimeofday(&end, NULL);
        for(int i=2; i>0; i--) { // shift the record
            e->host_time[i] = e->host_time[i-1];
        }
        e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        e->host_heat_score = 0;
        e->gdr_heat_score =  e->gdr_heat_score < 5 ? e->gdr_heat_score++ : 5;
    } else if(e->gdr_time[0] < 0) { // no record for gdr path, force to choose it
        gettimeofday(&start, NULL);
        ret = get_data_gdr(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
        gettimeofday(&end, NULL);
        for(int i=2; i>0; i--) { // shift the record
            e->gdr_time[i] = e->gdr_time[i-1];
        }
        e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        e->gdr_heat_score = 0;
        e->host_heat_score = e->host_heat_score < 5 ? e->host_heat_score++ : 5;
    } else { // have both records, choose the path according to score
        double host_perf_score, gdr_perf_score,
               host_total_score, gdr_total_score, max_total_score;
        double avg_host_time = 0.0, avg_gdr_time = 0.0;
        int avg_host_cnt = 0, avg_gdr_cnt = 0;
        double host_prob, gdr_prob;
        for(int i=0; i<3; i++) {
            if(e->host_time[i] > 0.0) {
                avg_host_time += e->host_time[i];
                avg_host_cnt++;
            }
            if(e->gdr_time[i] > 0.0) {
                avg_gdr_time += e->gdr_time[i];
                avg_gdr_cnt++;
            }
        }
        avg_host_time /= avg_host_cnt;
        avg_gdr_time /= avg_gdr_cnt;
        if(avg_gdr_time > avg_host_time) { // host perf better
            host_perf_score = 10;
            gdr_perf_score = 0;
        } else { // gdr perf better
            host_perf_score = 0;
            gdr_perf_score = 10;
        }
        host_total_score = host_perf_score + e->host_heat_score;
        gdr_total_score = gdr_perf_score + e->gdr_heat_score;
        max_total_score = host_total_score > gdr_total_score ? host_total_score : gdr_total_score;
        host_prob = exp(host_total_score - max_total_score) / (exp(host_total_score -max_total_score)
                                                            + exp(gdr_total_score -max_total_score));
        gdr_prob = exp(gdr_total_score - max_total_score) / (exp(host_total_score -max_total_score)
                                                            + exp(gdr_total_score -max_total_score));
        DEBUG_OUT("host_prob = %lf, gdr_prob = %lf\n", host_prob, gdr_prob);
        srand((unsigned)time(NULL));
        double r = ((double) rand() / (RAND_MAX));
        if(r < host_prob) { // choose host-based path
            gettimeofday(&start, NULL);
            ret = get_data_hybrid(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
            gettimeofday(&end, NULL);
            for(int i=2; i>0; i--) { // shift the record
                e->host_time[i] = e->host_time[i-1];
            }
            e->host_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->host_heat_score = 0;
            e->gdr_heat_score =  e->gdr_heat_score < 5 ? e->gdr_heat_score++ : 5;
        } else { // choose gdr path
            gettimeofday(&start, NULL);
            ret = get_data_gdr(client, num_odscs, req_obj, odsc_tab, d_data, ctime);
            gettimeofday(&end, NULL);
            for(int i=2; i>0; i--) { // shift the record
                e->gdr_time[i] = e->gdr_time[i-1];
            }
            e->gdr_time[0] = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
            e->gdr_heat_score = 0;
            e->host_heat_score = e->host_heat_score < 5 ? e->host_heat_score++ : 5;
        }
    }
    return ret;
}

static int get_data_dual_channel(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    /* 1 - Set a rdma size threshold according to ratio
       2 - split odsc tab according to the threshold && Margo_iforward()
       3 - Margo_wait() for all host path rpc && H->D transfer + ssd_copy_cuda
       4 - Margo_wait() for all gdr path rpc + ssd_copy_cuda
       5 - Tune the ratio according to the time 
    */
    int ret = dspaces_SUCCESS;
    struct timeval start, end;

    cudaError_t curet;
    struct cudaPointerAttributes ptr_attr;
    curet = cudaPointerGetAttributes(&ptr_attr, d_data);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                __func__, cudaGetErrorString(curet));
        return dspaces_ERR_CUDA;
    }

    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                    .device = ptr_attr.device};

    bulk_in_t *in = (bulk_in_t *)malloc(sizeof(bulk_in_t) * num_odscs);

    struct obj_data **od, **host_od, **device_od;
    od = (struct obj_data **) malloc(num_odscs * sizeof(struct obj_data *));

    margo_request *serv_req = 
        (margo_request *)malloc(sizeof(margo_request) * num_odscs);
    hg_handle_t *hndl = 
        (hg_handle_t *)malloc(sizeof(hg_handle_t) * num_odscs);

    size_t total_rdma_size = req_obj.size*bbox_volume(&req_obj.bb);

    // preset data volume for gdr / pipeline = 50% : 50%
    // cut the data byte stream and record the offset
    static double gdr_ratio = 0.5;
    static double host_ratio = 0.5;

    // first N odscs go to gdr: [0 : N-1]->gdr, [N: num_odsc-1]->host
    size_t gdr_rdma_size = 0;
    size_t rdma_size_threshold = (size_t) (total_rdma_size*gdr_ratio);
    int num_host, num_gdr = 0;
    for(int i=0; i<num_odscs; i++) {
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);
        hg_size_t rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);
        if(gdr_rdma_size < rdma_size_threshold) { // go to gdr 
            od[i] = obj_data_alloc_cuda(&odsc_tab[i]);
            margo_bulk_create_attr(client->mid, 1, (void **)(&(od[i]->data)),
                                &rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr,
                                &in[i].handle);
            gdr_rdma_size += rdma_size;
            num_gdr++;
        } else { // go to host
            od[i] = obj_data_alloc(&odsc_tab[i]);
            margo_bulk_create(client->mid, 1, (void **)(&(od[i]->data)),
                            &rdma_size, HG_BULK_WRITE_ONLY, &in[i].handle);
        }
        hg_addr_t server_addr;
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);
        hg_handle_t handle;
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &handle);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &handle);
        }
        margo_request req;
        // forward get requests
        margo_iforward(handle, &in[i], &req);
        hndl[i] = handle;
        serv_req[i] = req;
        margo_addr_free(client->mid, server_addr);  
    }

    num_host = num_odscs - num_gdr;

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    // concurrent cuda streams assigned to each od
    cudaStream_t *gdr_stream, *host_stream;
    int stream_size, gdr_stream_size, host_stream_size;
    
    if(num_odscs < client->cuda_info.num_concurrent_kernels) {
        stream_size = num_odscs;
        gdr_stream_size = num_gdr;
        host_stream_size = num_host;
    } else {
        stream_size = client->cuda_info.num_concurrent_kernels;
        gdr_stream_size = (int) ((1.0*num_gdr/num_odscs) * stream_size);
        host_stream_size = stream_size - gdr_stream_size;
    }

    gdr_stream = (cudaStream_t*) malloc(gdr_stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < gdr_stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&gdr_stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
            }
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }

    host_stream = (cudaStream_t*) malloc(host_stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < host_stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&host_stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
            }
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }
    

    device_od = malloc(num_host * sizeof(struct obj_data *));

    bulk_out_t resp;
    host_od = &od[num_gdr];

    // First, process the host-based path
    for(int i=0; i<num_host; i++) {
        device_od[i] = obj_data_alloc_cuda(&odsc_tab[num_gdr+i]);
        margo_wait(serv_req[num_gdr+i]);
        margo_get_output(hndl[num_gdr+i], &resp);
        // H->D async transfer
        size_t data_size = (req_obj.size) * bbox_volume(&odsc_tab[num_gdr+i].bb);
        curet = cudaMemcpyAsync(device_od[i]->data, host_od[i]->data, data_size,
                                    cudaMemcpyHostToDevice, host_stream[i%host_stream_size]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
            }
            for(int j = 0; i <= i; j++) {
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        ret = ssd_copy_cuda_async(return_od, device_od[i], &host_stream[i%host_stream_size]);
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%s)\n",
                    __func__, ret);
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
            }
            for(int j = 0; i <= i; j++) {
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        margo_free_output(hndl[i], &resp);
        margo_bulk_free(in[i].handle);
        margo_destroy(hndl[i]);
    }

    // Second, process the GDR path
    for(int i=0; i<num_gdr; i++) {
        margo_wait(serv_req[i]);
        margo_get_output(hndl[i], &resp);
        ret = ssd_copy_cuda_async(return_od, od[i], &gdr_stream[i%gdr_stream_size]);
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                    __func__, ret);
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        margo_free_output(hndl[i], &resp);
        margo_bulk_free(in[i].handle);
        margo_destroy(hndl[i]);
    }

    double gdr_timer = 0, host_timer = 0;
    double epsilon = 0.2; // 0.2ms
    /*  Try to tune the ratio every 2 timesteps
        At timestep (t), if 2nd timer(t) < 0.2ms, means 2nd request(t) finishes no later than the 1st(t).
            Keep the same ratio at (t+1), but swap the request.
            If the 2nd timer(t+1) < 0.2ms, means almost same time; else, tune the ratio and not swap request
            Suppose gdr finishes first initially: wait_flag = 0 -> host first; wait_flag = 1 -> gdr first
        else
    */
    cudaStream_t *stream0, *stream1;
    int stream_size0, stream_size1;
    double *timer0, *timer1;
    static int wait_flag = 0;
    if(wait_flag == 0) {
        stream0 = host_stream;
        stream_size0 = num_host;
        timer0 = &host_timer;
        stream1 = gdr_stream;
        stream_size1 = num_gdr;
        timer1 = &gdr_timer;
    } else {
        stream0 = gdr_stream;
        stream_size0 = num_gdr;
        timer0 = &gdr_timer;
        stream1 = host_stream;
        stream_size1 = num_host;
        timer1 = &host_timer;
    }

    for(int i=0; i<stream_size0; i++) {
        gettimeofday(&start, NULL);
        curet = cudaStreamSynchronize(stream0[i]);
        gettimeofday(&end, NULL);
        *timer0 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream0[i]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }

    for(int i=0; i<stream_size1; i++) {
        gettimeofday(&start, NULL);
        curet = cudaStreamSynchronize(stream1[i]);
        gettimeofday(&end, NULL);
        *timer1 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream1[i]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(gdr_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_gdr; j++) {
                obj_data_free_cuda(od[j]);
            }
            for(int j = 0; j < num_host; j++) {
                obj_data_free(od[num_gdr+j]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }

    free(host_stream);
    free(gdr_stream);

    if(*timer1 > stream_size1*epsilon) {
        // 2nd request takes longer time, tune ratio
        if(gdr_timer < host_timer) {
            if(host_timer - gdr_timer > 1e-3) {
                gdr_ratio += ((host_timer - gdr_timer) / host_timer) * (1-gdr_ratio);
                host_ratio = 1 - gdr_ratio;
            }
        } else {
            if(gdr_timer - host_timer > 1e-3) {
                gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * (gdr_ratio-0);
                host_ratio = 1 - gdr_ratio;
            }
        }
    } else {
        // 2nd request finishes no later than the 1st request
        // swap request by setting flag = 1
        wait_flag == 0 ? 1:0;
    }

    DEBUG_OUT("ts = %u, gdr_ratio = %lf, host_ratio = %lf,"
                "gdr_time = %lf, host_time = %lf\n", req_obj.version, gdr_ratio, host_ratio, 
                    gdr_timer, host_timer);
    
    for(int i = 0; i < num_gdr; i++) {
        obj_data_free_cuda(od[i]);
    }
    for(int i = 0; i < num_host; i++) {
        obj_data_free(od[num_gdr+i]);
        obj_data_free_cuda(device_od[i]);
    }
    free(device_od);
    free(od);
    free(hndl);
    free(serv_req);
    free(in);
    free(return_od);

    *ctime = 0;
    return ret;
}

static int get_data_hybrid_v2(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    int ret = dspaces_SUCCESS;
    hg_return_t hret = HG_SUCCESS;
    cudaError_t curet;
    struct timeval start, end;
    double timer = 0; // timer in second

    bulk_in_t *in =
        (bulk_in_t *) malloc(num_odscs*sizeof(bulk_in_t));
    struct obj_data **host_od =
        (struct obj_data**) malloc(num_odscs*sizeof(struct obj_data *));
    margo_request *serv_req =
        (margo_request *)malloc(num_odscs*sizeof(margo_request));
    hg_handle_t *hndl=
        (hg_handle_t *)malloc(num_odscs*sizeof(hg_handle_t));
    
    hg_size_t rdma_size;
    hg_addr_t server_addr;
    for(int i=0; i<num_odscs; i++) {
        host_od[i] = obj_data_alloc(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);
        margo_bulk_create(client->mid, 1, (void **)(&(host_od[i]->data)),
                          &rdma_size, HG_BULK_WRITE_ONLY, &in[i].handle);
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &hndl[i]);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &hndl[i]);
        }
        // forward get requests
        margo_iforward(hndl[i], &in[i], &serv_req[i]);
        margo_addr_free(client->mid, server_addr);
    }

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    int stream_size;
    if(num_odscs < client->cuda_info.num_concurrent_kernels) {
        stream_size = num_odscs;
    } else {
        stream_size = client->cuda_info.num_concurrent_kernels;
    }

    cudaStream_t* stream =
        (cudaStream_t*) malloc(stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i = 0; i < num_odscs; i++) {
                obj_data_free(host_od[i]);
            }
            free(host_od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }

    struct obj_data **device_od =
        (struct obj_data **) malloc(num_odscs * sizeof(struct obj_data *));
    bulk_out_t resp;
    struct list_head req_done_list;
    struct size_t_list_entry *e, *t;
    INIT_LIST_HEAD(&req_done_list);
    size_t h2d_size;
    size_t req_idx;
    do {
        hret = margo_wait_any(num_odscs, serv_req, &req_idx);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
                    __func__, req_idx, hret);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i=0; i < num_odscs; i++) {
                obj_data_free(host_od[i]);
            }
            free(host_od);
            free(in);
            return dspaces_ERR_MERCURY;
        }
        // break when all req are finished
        if(req_idx == num_odscs) {
            break;
        }
        serv_req[req_idx] = MARGO_REQUEST_NULL;
        margo_get_output(hndl[req_idx], &resp);
        device_od[req_idx] = obj_data_alloc_cuda(&odsc_tab[req_idx]);
        h2d_size = (req_obj.size) * bbox_volume(&odsc_tab[req_idx].bb);
        curet = cudaMemcpyAsync(device_od[req_idx]->data, host_od[req_idx]->data,
                                h2d_size, cudaMemcpyHostToDevice,
                                stream[req_idx%stream_size]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            obj_data_free_cuda(device_od[req_idx]);
            list_for_each_entry_safe(e, t, &req_done_list,
                                    struct size_t_list_entry, entry) {
                obj_data_free_cuda(device_od[e->value]);
                list_del(&e->entry);
                free(e);
            }
            free(device_od);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i=0; i < num_odscs; i++) {
                obj_data_free(host_od[i]);
            }
            free(host_od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        ret = ssd_copy_cuda_async(return_od, device_od[req_idx],
                                &stream[req_idx%stream_size]);
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                __func__, ret);
            obj_data_free_cuda(device_od[req_idx]);
            list_for_each_entry_safe(e, t, &req_done_list,
                                    struct size_t_list_entry, entry) {
                obj_data_free_cuda(device_od[e->value]);
                list_del(&e->entry);
                free(e);
            }
            free(device_od);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i=0; i < num_odscs; i++) {
                obj_data_free(host_od[i]);
            }
            free(host_od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        e = (struct size_t_list_entry *) malloc(sizeof(struct size_t_list_entry));
        e->value = req_idx;
        list_add(&e->entry, &req_done_list);
        margo_free_output(hndl[req_idx], &resp);
        margo_bulk_free(in[req_idx].handle);
        margo_destroy(hndl[req_idx]);
    } while(req_idx != num_odscs);

    // beyond this point, all device_od are allocated
    list_for_each_entry_safe(e, t, &req_done_list, struct size_t_list_entry, entry) {
        list_del(&e->entry);
        free(e);
    }

    for(int i=0; i < stream_size; i++) {
        curet = cudaStreamSynchronize(stream[i]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            for(int i = 0; i < num_odscs; i++) {
                obj_data_free_cuda(device_od[i]);
                obj_data_free(host_od[i]);
            }
            free(device_od);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            free(host_od);
            free(in);
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream[i]);
    }

    for(int i = 0; i < num_odscs; i++) {
        obj_data_free_cuda(device_od[i]);
        obj_data_free(host_od[i]);
    }
    free(device_od);
    free(hndl);
    free(serv_req);
    free(host_od);
    free(in);
    free(return_od);

    *ctime = timer;
    return ret;
}

static int get_data_gdr_v2(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    int ret = dspaces_SUCCESS;
    hg_return_t hret = HG_SUCCESS;
    struct timeval start, end;
    double timer = 0; // timer in second

    cudaError_t curet;
    struct cudaPointerAttributes ptr_attr;
    
    bulk_in_t *in =
        (bulk_in_t *) malloc(num_odscs*sizeof(bulk_in_t));
    struct obj_data **od =
        (struct obj_data**) malloc(num_odscs*sizeof(struct obj_data *));
    margo_request *serv_req =
        (margo_request *)malloc(num_odscs*sizeof(margo_request));
    hg_handle_t *hndl=
        (hg_handle_t *)malloc(num_odscs*sizeof(hg_handle_t));

    curet = cudaPointerGetAttributes(&ptr_attr, d_data);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                __func__, cudaGetErrorString(curet));
        free(serv_req);
        free(hndl);
        free(od);
        free(in);
        return dspaces_ERR_CUDA;
    }

    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                    .device = ptr_attr.device};

    hg_size_t rdma_size;
    hg_addr_t server_addr;
    for(int i = 0; i < num_odscs; ++i) {
        od[i] = obj_data_alloc_cuda(&odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);

        rdma_size = (req_obj.size) * bbox_volume(&odsc_tab[i].bb);

        margo_bulk_create_attr(client->mid, 1, (void **)(&(od[i]->data)),
                            &rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr,
                            &in[i].handle);
        margo_addr_lookup(client->mid, odsc_tab[i].owner, &server_addr);
        if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &hndl[i]);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &hndl[i]);
        }
        // forward get requests
        margo_iforward(hndl[i], &in[i], &serv_req[i]);
        margo_addr_free(client->mid, server_addr);
    }

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    // concurrent cuda streams assigned to each od
    
    int stream_size;
    if(num_odscs < client->cuda_info.num_concurrent_kernels) {
        stream_size = num_odscs;
    } else {
        stream_size = client->cuda_info.num_concurrent_kernels;
    }

    cudaStream_t* stream =
        (cudaStream_t*) malloc(stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i = 0; i < num_odscs; i++) {
                obj_data_free_cuda(od[i]);
            }
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
    }

    bulk_out_t resp;
    size_t req_idx;
    do {
        hret = margo_wait_any(num_odscs, serv_req, &req_idx);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
                    __func__, req_idx, hret);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i=0; i < num_odscs; i++) {
                obj_data_free_cuda(od[i]);
            }
            free(od);
            free(in);
            return dspaces_ERR_MERCURY;
        }
        // break when all req are finished
        if(req_idx == num_odscs) {
            break;
        }
        serv_req[req_idx] = MARGO_REQUEST_NULL;
        margo_get_output(hndl[req_idx], &resp);
        ret = ssd_copy_cuda_async(return_od, od[req_idx], 
                                    &stream[req_idx%stream_size]);
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr,
                "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                __func__, ret);
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i=0; i < num_odscs; i++) {
                obj_data_free_cuda(od[i]);
            }
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }
        margo_free_output(hndl[req_idx], &resp);
        margo_bulk_free(in[req_idx].handle);
        margo_destroy(hndl[req_idx]);
    } while(req_idx != num_odscs);

    for(int i=0; i < stream_size; i++) {
        curet = cudaStreamSynchronize(stream[i]);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int i = 0; i < num_odscs; i++) {
                obj_data_free_cuda(od[i]);
            }
            free(od);
            free(in);
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream[i]);
    }

    for(int i = 0; i < num_odscs; i++) {
        obj_data_free_cuda(od[i]);
    }

    free(hndl);
    free(serv_req);
    free(od);
    free(in);
    free(return_od);

    *ctime = timer;
    return ret;
}

static int get_data_dual_channel_v2(dspaces_client_t client, int num_odscs,
                    obj_descriptor req_obj, obj_descriptor *odsc_tab,
                    void *d_data, double *ctime)
{
    /* 1 - Set a rdma size threshold according to ratio
       2 - Split every odsc in odsc_tab into host-based & gdr
       3 - Margo_iforward() for host-based & gdr bulk I/O respectively
       4 - Margo_wait_any() for all bulk rpc && H->D transfer + ssd_copy_cuda_async
            or only ssd_copy_cuda_async
       5 - Tune the ratio according to the time 
    */

    // preset data volume for host_based / GDR = 50% : 50%
    // cut the data along the highest dimension
    // first piece goes to host, second piece goes to GDR
    static double host_ratio = 0.5;
    static double gdr_ratio = 0.5;
    // 1MB makes no difference for single or dual channel
    uint64_t ratio_eps = 1 << 20;
    // make it to pure GDR or host-based when either the ratio is around 1 
    double min_ratio = host_ratio > gdr_ratio ? gdr_ratio : host_ratio;
    uint64_t get_rdma_size = req_obj.size;
    for(int i=0; i<req_obj.bb.num_dims; i++) {
        get_rdma_size *= (req_obj.bb.ub.c[i] - req_obj.bb.lb.c[i] + 1);
    }
    if(min_ratio * get_rdma_size < ratio_eps) { // go to either pure GDR or host-based
        if(host_ratio > gdr_ratio) { // pure host
            return get_data_hybrid(client, num_odscs, req_obj,
                                    odsc_tab, d_data, ctime);
        } else { // pure GDR
            return get_data_gdr(client, num_odscs, req_obj,
                                    odsc_tab, d_data, ctime);
        }
    }

    int ret = dspaces_SUCCESS;
    hg_return_t hret = HG_SUCCESS;
    struct timeval start, end;

    cudaError_t curet;
    struct cudaPointerAttributes ptr_attr;

    curet = cudaPointerGetAttributes(&ptr_attr, d_data);
    if(curet != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                __func__, cudaGetErrorString(curet));
        return dspaces_ERR_CUDA;
    }

    struct hg_bulk_attr bulk_attr = {.mem_type = HG_MEM_TYPE_CUDA,
                                    .device = ptr_attr.device};

    int num_rpcs = 2*num_odscs;
    obj_descriptor *host_odsc_tab =
        (obj_descriptor*) malloc(num_odscs*sizeof(obj_descriptor));
    obj_descriptor *gdr_odsc_tab =
        (obj_descriptor*) malloc(num_odscs*sizeof(obj_descriptor));
    bulk_in_t *in =
        (bulk_in_t*) malloc(num_rpcs*sizeof(bulk_in_t));
    struct obj_data **od = 
        (struct obj_data**) malloc(num_rpcs*sizeof(struct obj_data*));
    margo_request *serv_req = 
        (margo_request*) malloc(num_rpcs*sizeof(margo_request));
    hg_handle_t *hndl = 
        (hg_handle_t*) malloc(num_rpcs*sizeof(hg_handle_t));

    memcpy(host_odsc_tab, odsc_tab, num_odscs*sizeof(obj_descriptor));
    memcpy(gdr_odsc_tab, odsc_tab, num_odscs*sizeof(obj_descriptor));

    int cut_dim; // find the highest dimension whose dim length > 1
    uint64_t dist; // the bbox distance of the cut_dim
    uint64_t cut_dist; // dist * host_ratio
    hg_size_t host_rdma_size, gdr_rdma_size;
    hg_addr_t server_addr;
    int gdr_idx;
    // first N elems in bulk_in, od, serv_req and hndl go to host:
    // [0 : N-1]->host, [N: num_odsc-1]->GDR
    for(int i=0; i<num_odscs; i++) {
        for(int j=0; j<odsc_tab[i].bb.num_dims; j++) {
            if(odsc_tab[i].bb.ub.c[j] - odsc_tab[i].bb.lb.c[j] > 0) {
                cut_dim = j;
                break;
            }
        }
        dist = odsc_tab[i].bb.ub.c[cut_dim]- odsc_tab[i].bb.lb.c[cut_dim] + 1;
        cut_dist  = dist * host_ratio;
        if(cut_dist == 0) { // host_ratio near zero, go to pure GDR
            for(int j = 0; j < j; j++) {
                margo_bulk_free(in[j].handle);
                margo_destroy(hndl[j]);
                obj_data_free(od[j]);
                margo_bulk_free(in[j+num_odscs].handle);
                margo_destroy(hndl[j+num_odscs]);
                obj_data_free_cuda(od[j+num_odscs]);
            }
            free(hndl);
            free(serv_req);
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return get_data_gdr(client, num_odscs, req_obj,
                                    odsc_tab, d_data, ctime);
        }
        host_odsc_tab[i].bb.ub.c[cut_dim] = 
            (uint64_t) (host_odsc_tab[i].bb.lb.c[cut_dim] + cut_dist - 1);
        margo_addr_lookup(client->mid, host_odsc_tab[i].owner, &server_addr);
        /* Start host-based RPC ASAP */
        od[i] = obj_data_alloc(&host_odsc_tab[i]);
        in[i].odsc.size = sizeof(obj_descriptor);
        in[i].odsc.raw_odsc = (char *)(&host_odsc_tab[i]);
        host_rdma_size = req_obj.size * bbox_volume(&host_odsc_tab[i].bb);
        margo_bulk_create(client->mid, 1, (void **)(&(od[i]->data)),
                            &host_rdma_size, HG_BULK_WRITE_ONLY, &in[i].handle);
        if(host_odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &hndl[i]);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &hndl[i]);
        }
        margo_iforward(hndl[i], &in[i], &serv_req[i]);
        /* GDR RPC */
        gdr_idx = i + num_odscs;
        if(cut_dist == dist) {
            gdr_odsc_tab[i].bb.lb.c[cut_dim] = host_odsc_tab[i].bb.ub.c[cut_dim];
        } else {
            gdr_odsc_tab[i].bb.lb.c[cut_dim] = host_odsc_tab[i].bb.ub.c[cut_dim] + 1;
        }
        od[gdr_idx] = obj_data_alloc_cuda(&gdr_odsc_tab[i]);
        in[gdr_idx].odsc.size = sizeof(obj_descriptor);
        in[gdr_idx].odsc.raw_odsc = (char *)(&gdr_odsc_tab[i]);
        gdr_rdma_size = req_obj.size * bbox_volume(&gdr_odsc_tab[i].bb);
        margo_bulk_create_attr(client->mid, 1, (void **)(&(od[gdr_idx]->data)),
                                &gdr_rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr,
                                &in[gdr_idx].handle);
        if(gdr_odsc_tab[i].flags & DS_CLIENT_STORAGE) {
            DEBUG_OUT("retrieving object from client-local storage.\n");
            margo_create(client->mid, server_addr, client->get_local_id,
                         &hndl[gdr_idx]);
        } else {
            DEBUG_OUT("retrieving object from server storage.\n");
            margo_create(client->mid, server_addr, client->get_id, &hndl[gdr_idx]);
        }
        margo_iforward(hndl[gdr_idx], &in[gdr_idx], &serv_req[gdr_idx]);
        margo_addr_free(client->mid, server_addr);
    }

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&req_obj, d_data);

    // concurrent cuda streams assigned to each od
    cudaStream_t *host_stream, *gdr_stream;
    int stream_size, host_stream_size, gdr_stream_size;
    
    if(num_rpcs < client->cuda_info.num_concurrent_kernels) {
        stream_size = num_rpcs;
    } else {
        stream_size = client->cuda_info.num_concurrent_kernels;     
    }
    host_stream_size = (int) (stream_size / 2);
    gdr_stream_size = stream_size - host_stream_size;

    host_stream = (cudaStream_t*) malloc(host_stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < host_stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&host_stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
            }
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_CUDA;
        }
    }

    gdr_stream = (cudaStream_t*) malloc(gdr_stream_size*sizeof(cudaStream_t));
    for(int i = 0; i < gdr_stream_size; i++) {
        curet = cudaStreamCreateWithFlags(&gdr_stream[i], cudaStreamNonBlocking);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            free(gdr_stream);
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
            }
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_CUDA;
        }
    }

    double host_timer = 0, gdr_timer = 0;
    
    struct obj_data **device_od =
        (struct obj_data **) malloc(num_odscs * sizeof(struct obj_data *));

    bulk_out_t resp;
    struct list_head req_done_list;
    struct size_t_list_entry *e, *t;
    INIT_LIST_HEAD(&req_done_list);
    size_t h2d_size;
    size_t req_idx;
    do {
        gettimeofday(&start, NULL);
        hret = margo_wait_any(num_rpcs, serv_req, &req_idx);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
                    __func__, req_idx, hret);
            list_for_each_entry_safe(e, t, &req_done_list,
                                     struct size_t_list_entry, entry) {
                obj_data_free_cuda(device_od[e->value]);
                list_del(&e->entry);
                free(e);
            }
            free(device_od);
            free(gdr_stream);
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
            }
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_MERCURY;
        }
        // break when all req are finished
        if(req_idx == num_rpcs) {
            break;
        }
        serv_req[req_idx] = MARGO_REQUEST_NULL;
        margo_get_output(hndl[req_idx], &resp);
        if(req_idx < num_odscs) { // host-based path
            device_od[req_idx] = obj_data_alloc_cuda(&host_odsc_tab[req_idx]);
            // H->D async transfer
            h2d_size = (req_obj.size) * bbox_volume(&host_odsc_tab[req_idx].bb);
            curet = cudaMemcpyAsync(device_od[req_idx]->data, od[req_idx]->data,
                                    h2d_size, cudaMemcpyHostToDevice,
                                    host_stream[req_idx%host_stream_size]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                obj_data_free_cuda(device_od[req_idx]);
                list_for_each_entry_safe(e, t, &req_done_list,
                                         struct size_t_list_entry, entry) {
                    obj_data_free_cuda(device_od[e->value]);
                    list_del(&e->entry);
                    free(e);
                }
                free(device_od);
                free(gdr_stream);
                free(host_stream);
                free(return_od);
                free(hndl);
                free(serv_req);
                for(int j = 0; j < num_odscs; j++) {
                    obj_data_free(od[j]);
                    obj_data_free_cuda(od[j+num_odscs]);
                }
                free(od);
                free(in);
                free(gdr_odsc_tab);
                free(host_odsc_tab);
                return dspaces_ERR_CUDA;
            }
            // track allocated device_od
            e = (struct size_t_list_entry *) malloc(sizeof(struct size_t_list_entry));
            e->value = req_idx;
            list_add(&e->entry, &req_done_list);
            ret = ssd_copy_cuda_async(return_od, device_od[req_idx],
                                &host_stream[req_idx%host_stream_size]);
            gettimeofday(&end, NULL);
            host_timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        } else { // GDR path
            ret = ssd_copy_cuda_async(return_od, od[req_idx], 
                            &gdr_stream[(req_idx-num_odscs)%gdr_stream_size]);
            gettimeofday(&end, NULL);
            gdr_timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        }
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                __func__, ret);
            list_for_each_entry_safe(e, t, &req_done_list, struct size_t_list_entry, entry) {
                obj_data_free_cuda(device_od[e->value]);
                list_del(&e->entry);
                free(e);
            }
            free(device_od);
            free(gdr_stream);
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
            }
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_CUDA;
        }
        margo_free_output(hndl[req_idx], &resp);
        margo_bulk_free(in[req_idx].handle);
        margo_destroy(hndl[req_idx]);
    } while(req_idx != num_rpcs);

    // beyond this point, all device_od are allocated
    list_for_each_entry_safe(e, t, &req_done_list, struct size_t_list_entry, entry) {
        list_del(&e->entry);
        free(e);
    }

    /*  Try to tune the ratio every 2 timesteps
        At timestep (t), if 2nd timer(t) < 0.2ms, means 2nd request(t) finishes no later than the 1st(t).
            Keep the same ratio at (t+1), but swap the request.
            If the 2nd timer(t+1) < 0.2ms, means almost same time; else, tune the ratio and not swap request
            Suppose gdr finishes first initially: wait_flag = 0 -> GDR first; wait_flag = 1 -> host first
        else
    */

    cudaStream_t *stream0, *stream1;
    int stream_size0, stream_size1;
    double *timer0, *timer1;
    static int wait_flag = 0;
    if(wait_flag == 0) {
        stream0 = gdr_stream;
        stream_size0 = gdr_stream_size;
        timer0 = &gdr_timer;
        stream1 = host_stream;
        stream_size1 = host_stream_size;
        timer1 = &host_timer;
    } else {
        stream0 = host_stream;
        stream_size0 = host_stream_size;
        timer0 = &host_timer;
        stream1 = gdr_stream;
        stream_size1 = gdr_stream_size;
        timer1 = &gdr_timer;
    }

    for(int i=0; i<stream_size0; i++) {
        gettimeofday(&start, NULL);
        curet = cudaStreamSynchronize(stream0[i]);
        gettimeofday(&end, NULL);
        *timer0 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(gdr_stream);
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream0[i]);
    }

    for(int i=0; i<stream_size1; i++) {
        gettimeofday(&start, NULL);
        curet = cudaStreamSynchronize(stream1[i]);
        gettimeofday(&end, NULL);
        *timer1 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
            free(gdr_stream);
            free(host_stream);
            free(return_od);
            free(hndl);
            free(serv_req);
            for(int j = 0; j < num_odscs; j++) {
                obj_data_free(od[j]);
                obj_data_free_cuda(od[j+num_odscs]);
                obj_data_free_cuda(device_od[j]);
            }
            free(device_od);
            free(od);
            free(in);
            free(gdr_odsc_tab);
            free(host_odsc_tab);
            return dspaces_ERR_CUDA;
            return dspaces_ERR_CUDA;
        }

        curet = cudaStreamDestroy(stream1[i]);
    }

    free(gdr_stream);
    free(host_stream);
    
    double stream_epsilon = 0.2; // 0.2ms

    if(*timer1 > stream_size1*stream_epsilon) {
        // 2nd request takes longer time, tune ratio
        if(gdr_timer < host_timer) {
            if(host_timer - gdr_timer > 1e-3) {
                gdr_ratio += ((host_timer - gdr_timer) / host_timer) * (1-gdr_ratio);
                host_ratio = 1 - gdr_ratio;
            }
        } else {
            if(gdr_timer - host_timer > 1e-3) {
                gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * (gdr_ratio-0);
                host_ratio = 1 - gdr_ratio;
            }
        }
    } else {
        // 2nd request finishes no later than the 1st request
        // swap request by setting flag = 1
        wait_flag == 0 ? 1:0;
    }

    DEBUG_OUT("ts = %u, gdr_ratio = %lf, host_ratio = %lf,"
                "gdr_time = %lf, host_time = %lf\n", req_obj.version, gdr_ratio, host_ratio, 
                    gdr_timer, host_timer);
    
    for(int j = 0; j < num_odscs; j++) {
        obj_data_free(od[j]);
        obj_data_free_cuda(od[j+num_odscs]);
        obj_data_free_cuda(device_od[j]);
    }
    free(device_od);
    free(od);
    free(hndl);
    free(serv_req);
    free(in);
    free(gdr_odsc_tab);
    free(host_odsc_tab);
    free(return_od);

    *ctime = 0;
    return ret;
}

static int dspaces_cuda_dcds_get(dspaces_client_t client, const char *var_name, unsigned int ver,
                     int elem_size, int ndim, uint64_t *lb, uint64_t *ub, void *data,
                     int timeout, double* ttime, double* ctime)
{
    /*
        1 - Query odscs
        2 - During the query, check the pattern. 
            Sub the predicted data objs if the pattern exists.
        3 - Check the local storage to see if there is any od
            included by the queried odsc; od in local storage
            should has exactly the same bbox as the queired result
        4 - H->D transfer for local od, and start ssd_copy_cuda_async    
        5 - wait odscs query
    */

    // preset data volume for host_based / GDR = 50% : 50%
    // cut the data along the highest dimension
    // first piece goes to host, second piece goes to GDR
    static double host_ratio = 0.5;
    static double gdr_ratio = 0.5;
    // 1MB makes no difference for single or dual channel
    uint64_t ratio_eps = 1 << 20;
    // make it to pure GDR or host-based when either the ratio is around 1 
    double min_ratio = host_ratio > gdr_ratio ? gdr_ratio : host_ratio;
    uint64_t get_rdma_size = elem_size;
    for(int i=0; i<ndim; i++) {
        get_rdma_size *= (ub[i] - lb[i] + 1);
    }
    int pure_host = 0, pure_gdr = 0;
    if(min_ratio * get_rdma_size < ratio_eps) {
        if(host_ratio > gdr_ratio) {
            pure_host = 1;
        } else {
            pure_gdr = 1;
        }
    }

    int ret = dspaces_SUCCESS;
    cudaError_t curet;
    hg_return_t hret;
    struct timeval start, end;
    int alloc_cnt = 0;
    struct cudaPointerAttributes ptr_attr;

    hg_addr_t my_addr, server_addr;
    size_t owner_addr_size = 128;
    
    hg_handle_t qhandle, sub_handle;
    margo_request qreq, sub_req;
    odsc_gdim_t qin, sub_in;
    odsc_list_t qout;

    int init_pattern;
    obj_descriptor qodsc, sub_odsc;
    struct global_dimension qod_gdim;
    struct subods_list_entry *subod_ent;

    fill_odsc(var_name, ver, elem_size, ndim, lb, ub, &qodsc);
    set_global_dimension(&(client->dcg->gdim_list), qodsc.name,
                         &(client->dcg->default_gdim), &qod_gdim);
    
    qin.odsc_gdim.size = sizeof(qodsc);
    qin.odsc_gdim.raw_odsc = (char *)(&qodsc);
    qin.odsc_gdim.gdim_size = sizeof(qod_gdim);
    qin.odsc_gdim.raw_gdim = (char *)(&qod_gdim);
    qin.param = timeout;

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->query_id, &qhandle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_create() failed with %d.\n", __func__,
                hret);
        margo_addr_free(client->mid, server_addr);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_iforward(qhandle, &qin, &qreq);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_iforward() failed with %d.\n",
                __func__, hret);
        margo_destroy(qhandle);
        margo_addr_free(client->mid, server_addr);
        return dspaces_ERR_MERCURY;
    }

    struct hg_bulk_attr bulk_attr;
    if(!pure_host) {
        curet = cudaPointerGetAttributes(&ptr_attr, data);
        if(curet != cudaSuccess) {
            fprintf(stderr, "ERROR: (%s): cudaPointerGetAttributes() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
            margo_destroy(qhandle);
            margo_addr_free(client->mid, server_addr);
            return dspaces_ERR_CUDA;
        }
        bulk_attr.mem_type = HG_MEM_TYPE_CUDA;
        bulk_attr.device = ptr_attr.device;
    }

    if(client->listener_init == 0) {
        ret = dspaces_init_listener(client);
        if(ret != dspaces_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): dspaces_init_listener() failed, Err Code: (%s)\n",
                    __func__, ret);
            margo_destroy(qhandle);
            margo_addr_free(client->mid, server_addr);
            return (ret);
        }
    }
    // check get obj pattern
    struct getobj_list_entry *getobj_ent =
        lookup_getobj_list(&client->dcg->getobj_record_list, qodsc);
    if(getobj_ent) {
        init_pattern = 0;
        if(getobj_ent->interval_ver != (qodsc.version-getobj_ent->last_ver)) { // keep pattern updated
            getobj_ent->interval_ver = (qodsc.version-getobj_ent->last_ver);
        }
        // sub the next req_obj (version = req_obj.version+ getobj_ent->interval_ver)
        memcpy(&sub_odsc, &qodsc, sizeof(obj_descriptor));
        sub_odsc.version += getobj_ent->interval_ver;
        // A hack to send our address to the server without using more space. This
        // field is ignored in a normal query.
        margo_addr_self(client->mid, &my_addr);
        margo_addr_to_string(client->mid, sub_odsc.owner, &owner_addr_size,
                            my_addr);
        margo_addr_free(client->mid, my_addr);
        sub_in.odsc_gdim.size = sizeof(sub_odsc);
        sub_in.odsc_gdim.raw_odsc = (char *)(&sub_odsc);
        sub_in.odsc_gdim.gdim_size = sizeof(qod_gdim);
        sub_in.odsc_gdim.raw_gdim = (char *)(&qod_gdim);
        // use sub_ods_serial as the key in the list
        sub_in.param = client->sub_ods_serial;
        hret = margo_create(client->mid, server_addr, client->sub_ods_id, &sub_handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: %s: margo_create() failed with %d.\n", __func__,
                    hret);
            margo_destroy(qhandle);
            margo_addr_free(client->mid, server_addr);
            return dspaces_ERR_SUB;
        }
        subod_ent = (struct subods_list_entry*) malloc(sizeof(*subod_ent));
        memcpy(&subod_ent->qodsc, &sub_odsc, sizeof(obj_descriptor));
        ABT_mutex_lock(client->sub_ods_mutex);
        subod_ent->id = client->sub_ods_serial++;
        list_add(&subod_ent->entry, &client->dcg->sub_ods_list);
        subod_ent->status = DSPACES_SUB_WAIT;
        ABT_mutex_unlock(client->sub_ods_mutex);
        hret = margo_iforward(sub_handle, &sub_in, &sub_req);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: %s: margo_iforward() failed with %d.\n",
                    __func__, hret);
            ABT_mutex_lock(client->sub_ods_mutex);
            list_del(&subod_ent->entry);
            free(subod_ent);
            ABT_mutex_unlock(client->sub_ods_mutex);
            margo_destroy(sub_handle);
            margo_destroy(qhandle);
            margo_addr_free(client->mid, server_addr);
            return dspaces_ERR_SUB;
        }
        getobj_ent->last_ver = qodsc.version;
        margo_destroy(sub_handle);
    } else {
        init_pattern = 1;
        getobj_ent = (struct getobj_list_entry*) malloc(sizeof(struct getobj_list_entry));
        getobj_ent->var_name= strdup(qodsc.name);
        getobj_ent->st = qodsc.st;
        memcpy(&getobj_ent->bb, &qodsc.bb, sizeof(struct bbox));
        getobj_ent->last_ver = qodsc.version;
        getobj_ent->interval_ver = 0;
        list_add(&getobj_ent->entry, &client->dcg->getobj_record_list);
    }
    margo_addr_free(client->mid, server_addr);

    // return_od is linked to the device ptr
    struct obj_data *return_od = obj_data_alloc_no_data(&qodsc, data);

    // check local storage, the local od we can find
    // could only be a subset of queried odsc list
    struct obj_data **local_od;
    ABT_mutex_lock(client->ls_mutex);
    int num_local = ls_find_ods_include(client->dcg->ls, &qodsc, &local_od);
    ABT_mutex_unlock(client->ls_mutex);

    int local_stream_size;
    cudaStream_t *local_stream;
    struct obj_data **local_device_od;
    size_t h2d_size;
    if(num_local) {
        // the local od we can find could only be a subset of queried odsc list
        // D->H transfer and ssd_copy_async for local od
        if(num_local < client->cuda_info.num_concurrent_kernels) {
            local_stream_size = num_local;
        } else {
            local_stream_size = client->cuda_info.num_concurrent_kernels;
        }
        local_stream = (cudaStream_t*) malloc(local_stream_size*sizeof(cudaStream_t));
        alloc_cnt = 0;
        for(int i=0; i<local_stream_size; i++) {
            curet = cudaStreamCreateWithFlags(&local_stream[i], cudaStreamNonBlocking);
            if(curet != cudaSuccess) {
                fprintf(stderr, 
                    "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                    __func__, cudaGetErrorString(curet));
                for(int j=0; j<alloc_cnt; j++) {
                    cudaStreamDestroy(local_stream[j]);
                }
                free(local_stream);
                free(local_od);
                free(return_od);
                if(init_pattern) {
                    list_del(&getobj_ent->entry);
                    free(getobj_ent->var_name);
                    free(getobj_ent);
                }
                margo_destroy(qhandle);
                return dspaces_ERR_CUDA;
            }
            alloc_cnt++ ;
        }

        local_device_od = 
            (struct obj_data **) malloc(num_local * sizeof(struct obj_data *));

        for(int i=0; i<num_local; i++) {
            local_device_od[i] = obj_data_alloc_cuda(&local_od[i]->obj_desc);
            h2d_size = qodsc.size * bbox_volume(&local_od[i]->obj_desc.bb);
            curet = cudaMemcpyAsync(local_device_od[i]->data, local_od[i]->data,
                                    h2d_size, cudaMemcpyHostToDevice,
                                    local_stream[i%local_stream_size]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                        __func__, cudaGetErrorString(curet));
                obj_data_free_cuda(local_device_od[i]);
                for(int j=0; j<alloc_cnt; j++) {
                    obj_data_free_cuda(local_device_od[j]);
                }
                free(local_device_od);
                for(int j=0; j<local_stream_size; j++) {
                    cudaStreamDestroy(local_stream[j]);
                }
                free(local_stream);
                free(local_od);
                free(return_od);
                if(init_pattern) {
                    list_del(&getobj_ent->entry);
                    free(getobj_ent->var_name);
                    free(getobj_ent);
                }
                margo_destroy(qhandle);
                return dspaces_ERR_CUDA;
            }
            ret = ssd_copy_cuda_async(return_od, local_device_od[i],
                                        &local_stream[i%local_stream_size]);
            if(ret != dspaces_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%s)\n",
                        __func__, ret);
                obj_data_free_cuda(local_device_od[i]);
                for(int j=0; j<alloc_cnt; j++) {
                    obj_data_free_cuda(local_device_od[j]);
                }
                free(local_device_od);
                for(int j=0; j<local_stream_size; j++) {
                    cudaStreamDestroy(local_stream[j]);
                }
                free(local_stream);
                free(local_od);
                free(return_od);
                if(init_pattern) {
                    list_del(&getobj_ent->entry);
                    free(getobj_ent->var_name);
                    free(getobj_ent);
                }
                margo_destroy(qhandle);
                return dspaces_ERR_CUDA;
            }
            alloc_cnt++;
        }
    }
    
    // wait odsc query
    hret = margo_wait(qreq);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_wait() failed with %d.\n",
                __func__, hret);
        for(int i=0; i<num_local; i++) {
            obj_data_free_cuda(local_device_od[i]);
        }
        free(local_device_od);
        for(int i=0; i<local_stream_size; i++) {
            cudaStreamDestroy(local_stream[i]);
        }
        free(local_stream);
        free(local_od);
        free(return_od);
        if(init_pattern) {
            list_del(&getobj_ent->entry);
            free(getobj_ent->var_name);
            free(getobj_ent);
        }
        margo_destroy(qhandle);
        return dspaces_ERR_MERCURY;
    }
    hret = margo_get_output(qhandle, &qout);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_get_output() failed with %d.\n",
                __func__, hret);
        for(int i=0; i<num_local; i++) {
            obj_data_free_cuda(local_device_od[i]);
        }
        free(local_device_od);
        for(int i=0; i<local_stream_size; i++) {
            cudaStreamDestroy(local_stream[i]);
        }
        free(local_stream);
        free(local_od);
        free(return_od);
        if(init_pattern) {
            list_del(&getobj_ent->entry);
            free(getobj_ent->var_name);
            free(getobj_ent);
        }
        margo_destroy(qhandle);
        return (0);
    }

    int num_odscs = (qout.odsc_list.size) / sizeof(obj_descriptor);
    obj_descriptor *odsc_tab = (obj_descriptor*) malloc(qout.odsc_list.size);
    memcpy(odsc_tab, qout.odsc_list.raw_odsc, qout.odsc_list.size);
    
    // Pick up remote odscs from odsc_tab
    obj_descriptor *remote_odsc_tab = 
        (obj_descriptor*) malloc(num_odscs*sizeof(obj_descriptor));
    int num_remote = 0;
    int local_found;
    if(num_local) {
        for(int i=0; i<num_odscs; i++) {
            // check if local_od_tab has this odsc
            local_found = 0;
            for(int j=0; j<num_local; j++) {
                if(obj_desc_equals_no_owner(&odsc_tab[i],
                                &local_od[j]->obj_desc)) {
                    local_found = 1;
                    break;
                }
            }
            // go to remote tab
            if(!local_found) {
                memcpy(&remote_odsc_tab[num_remote++], &odsc_tab[i], sizeof(obj_descriptor));
            }   
        }
        *ctime = num_local / num_odscs;
    } else {
        num_remote = num_odscs;
        memcpy(remote_odsc_tab, qout.odsc_list.raw_odsc, qout.odsc_list.size);
        *ctime = 0;
    }
    

    int num_rpcs;
    struct obj_data **remote_od;
    bulk_in_t *bin;
    margo_request *breq;
    hg_handle_t *bhndl;

    double gdr_timer = 0, host_timer = 0;

    int cut_dim; // find the highest dimension whose dim length > 1
    uint64_t dist; // the bbox distance of the cut_dim
    uint64_t cut_dist; // dist * host_ratio
    hg_size_t host_rdma_size, gdr_rdma_size;
    hg_addr_t bserver_addr;
    int gdr_idx;

    cudaStream_t *host_stream, *gdr_stream, *remote_stream;
    int remote_stream_size, host_stream_size, gdr_stream_size;

    struct obj_data **remote_device_od;
    bulk_out_t bresp;
    struct list_head req_done_list;
    struct size_t_list_entry *req_ent, *req_tmp;
    size_t req_idx;

    if(num_remote) {
        if(pure_host || pure_gdr) {
            remote_od = (struct obj_data**) malloc(num_remote*sizeof(struct obj_data*));
            bin = (bulk_in_t *)malloc(num_remote*sizeof(bulk_in_t));
            breq = (margo_request *)malloc(num_remote*sizeof(margo_request));
            bhndl = (hg_handle_t *)malloc(num_remote*sizeof(hg_handle_t));
            if(pure_host) { // pure host
                for(int i=0; i<num_remote; i++) {
                    remote_od[i] = obj_data_alloc(&remote_odsc_tab[i]);
                    bin[i].odsc.size = sizeof(obj_descriptor);
                    bin[i].odsc.raw_odsc = (char *)(&remote_odsc_tab[i]);
                    host_rdma_size = qodsc.size * bbox_volume(&remote_odsc_tab[i].bb);
                    margo_bulk_create(client->mid, 1, (void **)(&(remote_od[i]->data)),
                                        &host_rdma_size, HG_BULK_WRITE_ONLY, &bin[i].handle);
                    margo_addr_lookup(client->mid, remote_odsc_tab[i].owner, &bserver_addr);
                    if(remote_odsc_tab[i].flags & DS_CLIENT_STORAGE) {
                        DEBUG_OUT("retrieving object from client-local storage.\n");
                        margo_create(client->mid, bserver_addr, client->get_local_id,
                                    &bhndl[i]);
                    } else {
                        DEBUG_OUT("retrieving object from server storage.\n");
                        margo_create(client->mid, bserver_addr, client->get_id, &bhndl[i]);
                    }
                    margo_iforward(bhndl[i], &bin[i], &breq[i]);
                    margo_addr_free(client->mid, bserver_addr);
                }
            } else { // pure GDR
                for(int i=0; i<num_remote; i++) {
                    remote_od[i] = obj_data_alloc_cuda(&remote_odsc_tab[i]);
                    bin[i].odsc.size = sizeof(obj_descriptor);
                    bin[i].odsc.raw_odsc = (char *)(&remote_odsc_tab[i]);
                    gdr_rdma_size = qodsc.size * bbox_volume(&remote_odsc_tab[i].bb);
                    margo_bulk_create_attr(client->mid, 1, (void **)(&(remote_od[i]->data)),
                                            &gdr_rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr,
                                            &bin[i].handle);
                    margo_addr_lookup(client->mid, remote_odsc_tab[i].owner, &bserver_addr);
                    if(remote_odsc_tab[i].flags & DS_CLIENT_STORAGE) {
                        DEBUG_OUT("retrieving object from client-local storage.\n");
                        margo_create(client->mid, bserver_addr, client->get_local_id,
                                    &bhndl[i]);
                    } else {
                        DEBUG_OUT("retrieving object from server storage.\n");
                        margo_create(client->mid, bserver_addr, client->get_id, &bhndl[i]);
                    }
                    margo_iforward(bhndl[i], &bin[i], &breq[i]);
                    margo_addr_free(client->mid, bserver_addr);
                }
            }
            free(odsc_tab);
            // CUDA stream allocation
            if(num_remote < client->cuda_info.num_concurrent_kernels) {
                remote_stream_size = num_remote;
            } else {
                remote_stream_size = client->cuda_info.num_concurrent_kernels;
            }
            remote_stream = (cudaStream_t*) malloc(remote_stream_size*sizeof(cudaStream_t));
            alloc_cnt = 0;
            for(int i=0; i<remote_stream_size; i++) {
                curet = cudaStreamCreateWithFlags(&remote_stream[i], cudaStreamNonBlocking);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    for(int j=0; j<alloc_cnt; j++) {
                        cudaStreamDestroy(remote_stream[j]);
                    }
                    free(remote_stream);
                    if(pure_host) {
                        for(int j=0; j<num_remote; j++) {
                            obj_data_free(remote_od[j]);
                        }
                    } else {
                        for(int j=0; j<num_remote; j++) {
                            obj_data_free_cuda(remote_od[j]);
                        }
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    for(int j=0; j<num_local; j++) {
                        obj_data_free_cuda(local_device_od[j]);
                    }
                    free(local_device_od);
                    for(int j=0; j<local_stream_size; j++) {
                        cudaStreamDestroy(local_stream[j]);
                    }
                    free(local_stream);
                    free(local_od);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                alloc_cnt++ ;
            }
        } else { // dual-channel
            // RPC resources allocation
            num_rpcs = 2*num_remote;
            remote_odsc_tab = 
                (obj_descriptor*) realloc(remote_odsc_tab, num_rpcs*sizeof(obj_descriptor));
            remote_od = (struct obj_data**) malloc(num_rpcs*sizeof(struct obj_data*));
            bin = (bulk_in_t *)malloc(num_rpcs*sizeof(bulk_in_t));
            breq = (margo_request *)malloc(num_rpcs*sizeof(margo_request));
            bhndl = (hg_handle_t *)malloc(num_rpcs*sizeof(hg_handle_t));
            memcpy(&remote_odsc_tab[num_remote], remote_odsc_tab, num_remote*sizeof(obj_descriptor));

            // first N elems in remote_odsc_tab, bin, remote_od, breq and bhndl
            // go to host: [0 : N-1]->host, [N: num_odsc-1]->GDR
            for(int i=0; i<num_remote; i++) {
                for(int j=0; j<remote_odsc_tab[i].bb.num_dims; j++) {
                    if(remote_odsc_tab[i].bb.ub.c[j] - remote_odsc_tab[i].bb.lb.c[j] > 0) {
                        cut_dim = j;
                        break;
                    }
                }
                dist = remote_odsc_tab[i].bb.ub.c[cut_dim]- remote_odsc_tab[i].bb.lb.c[cut_dim] + 1;
                cut_dist = dist * host_ratio;
                remote_odsc_tab[i].bb.ub.c[cut_dim] = 
                    (uint64_t) (remote_odsc_tab[i].bb.lb.c[cut_dim] + cut_dist -1);
                margo_addr_lookup(client->mid, remote_odsc_tab[i].owner, &bserver_addr);
                /* Start host-based RPC ASAP */
                remote_od[i] = obj_data_alloc(&remote_odsc_tab[i]);
                bin[i].odsc.size = sizeof(obj_descriptor);
                bin[i].odsc.raw_odsc = (char *)(&remote_odsc_tab[i]);
                host_rdma_size = qodsc.size * bbox_volume(&remote_odsc_tab[i].bb);
                margo_bulk_create(client->mid, 1, (void **)(&(remote_od[i]->data)),
                                    &host_rdma_size, HG_BULK_WRITE_ONLY, &bin[i].handle);
                if(remote_odsc_tab[i].flags & DS_CLIENT_STORAGE) {
                    DEBUG_OUT("retrieving object from client-local storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_local_id,
                                &bhndl[i]);
                } else {
                    DEBUG_OUT("retrieving object from server storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_id, &bhndl[i]);
                }
                margo_iforward(bhndl[i], &bin[i], &breq[i]);
                /* GDR RPC */
                gdr_idx = i + num_remote;
                if(cut_dist == dist) {
                    remote_odsc_tab[gdr_idx].bb.lb.c[cut_dim] = remote_odsc_tab[i].bb.ub.c[cut_dim];
                } else {
                    remote_odsc_tab[gdr_idx].bb.lb.c[cut_dim] = remote_odsc_tab[i].bb.ub.c[cut_dim] + 1;
                }
                remote_od[gdr_idx] = obj_data_alloc_cuda(&remote_odsc_tab[gdr_idx]);
                bin[gdr_idx].odsc.size = sizeof(obj_descriptor);
                bin[gdr_idx].odsc.raw_odsc = (char *)(&remote_odsc_tab[gdr_idx]);
                gdr_rdma_size = qodsc.size * bbox_volume(&remote_odsc_tab[gdr_idx].bb);
                margo_bulk_create_attr(client->mid, 1, (void **)(&(remote_od[gdr_idx]->data)),
                                        &gdr_rdma_size, HG_BULK_WRITE_ONLY, &bulk_attr,
                                        &bin[gdr_idx].handle);
                if(remote_odsc_tab[gdr_idx].flags & DS_CLIENT_STORAGE) {
                    DEBUG_OUT("retrieving object from client-local storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_local_id,
                                &bhndl[gdr_idx]);
                } else {
                    DEBUG_OUT("retrieving object from server storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_id, &bhndl[gdr_idx]);
                }
                margo_iforward(bhndl[gdr_idx], &bin[gdr_idx], &breq[gdr_idx]);
                margo_addr_free(client->mid, bserver_addr);
            }

            free(odsc_tab);

            // CUDA stream allocation
            if(num_rpcs < client->cuda_info.num_concurrent_kernels) {
                remote_stream_size = num_rpcs;
            } else {
                remote_stream_size = client->cuda_info.num_concurrent_kernels;
            }
            host_stream_size = (int) (remote_stream_size / 2);
            gdr_stream_size = remote_stream_size - host_stream_size;

            host_stream = (cudaStream_t*) malloc(host_stream_size*sizeof(cudaStream_t));
            alloc_cnt = 0;
            for(int i = 0; i < host_stream_size; i++) {
                curet = cudaStreamCreateWithFlags(&host_stream[i], cudaStreamNonBlocking);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    for(int j=0; j<alloc_cnt; j++) {
                        cudaStreamDestroy(host_stream[j]);
                    }
                    free(host_stream);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    for(int j=0; j<num_local; j++) {
                        obj_data_free_cuda(local_device_od[j]);
                    }
                    free(local_device_od);
                    for(int j=0; j<local_stream_size; j++) {
                        cudaStreamDestroy(local_stream[j]);
                    }
                    free(local_stream);
                    free(local_od);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                alloc_cnt++ ;
            }

            gdr_stream = (cudaStream_t*) malloc(gdr_stream_size*sizeof(cudaStream_t));
            alloc_cnt = 0;
            for(int i = 0; i < gdr_stream_size; i++) {
                curet = cudaStreamCreateWithFlags(&gdr_stream[i], cudaStreamNonBlocking);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaStreamCreateWithFlags() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    for(int j=0; j<alloc_cnt; j++) {
                        cudaStreamDestroy(gdr_stream[j]);
                    }
                    free(gdr_stream);
                    for(int j=0; j<host_stream_size; j++) {
                        cudaStreamDestroy(host_stream[j]);
                    }
                    free(host_stream);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    for(int j=0; j<num_local; j++) {
                        obj_data_free_cuda(local_device_od[j]);
                    }
                    free(local_device_od);
                    for(int j=0; j<local_stream_size; j++) {
                        cudaStreamDestroy(local_stream[j]);
                    }
                    free(local_stream);
                    free(local_od);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                alloc_cnt++ ;
            } 
        }
        if(!pure_gdr) {
            remote_device_od = 
                (struct obj_data **) malloc(num_remote * sizeof(struct obj_data*));
            INIT_LIST_HEAD(&req_done_list);
        }
    }
    
    // free odsc query resources
    margo_free_output(qhandle, &qout);
    margo_destroy(qhandle);

    if(num_local) {
        // finish ssd_copy_cuda_async of local od
        for(int i=0; i<local_stream_size; i++) {
            curet = cudaStreamSynchronize(local_stream[i]);
            if(curet != cudaSuccess) {
                fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                if(num_remote) {
                    free(remote_device_od);
                    for(int j=0; j<host_stream_size; j++) {
                        cudaStreamDestroy(host_stream[j]);
                    }
                    free(host_stream);
                    for(int j=0; j<gdr_stream_size; j++) {
                        cudaStreamDestroy(gdr_stream[j]);
                    }
                    free(gdr_stream);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                }
                free(remote_odsc_tab);
                for(int j=0; j<num_local; j++) {
                    obj_data_free_cuda(local_device_od[j]);
                }
                free(local_device_od);
                for(int j=i; j<local_stream_size; j++) {
                    cudaStreamDestroy(local_stream[j]);
                }
                free(local_stream);
                free(local_od);
                free(return_od);
                if(init_pattern) {
                    list_del(&getobj_ent->entry);
                    free(getobj_ent->var_name);
                    free(getobj_ent);
                }
                return dspaces_ERR_CUDA;
            }
            cudaStreamDestroy(local_stream[i]);
        }

        ABT_mutex_lock(client->ls_mutex);
        for(int i=0; i<num_local; i++) {
            obj_data_free_cuda(local_device_od[i]);
            ls_remove(client->dcg->ls, local_od[i]);
            obj_data_free(local_od[i]);
        }
        ABT_mutex_unlock(client->ls_mutex);
        free(local_device_od);
        free(local_stream);
        free(local_od);
    }
    
    /*  Try to tune the ratio every 2 timesteps
        At timestep (t), if 2nd timer(t) < 0.2ms, means 2nd request(t) finishes no later than the 1st(t).
            Keep the same ratio at (t+1), but swap the request.
            If the 2nd timer(t+1) < 0.2ms, means almost same time; else, tune the ratio and not swap request
            Suppose gdr finishes first initially: wait_flag = 0 -> GDR first; wait_flag = 1 -> host first
        else
    */
    cudaStream_t *stream0, *stream1;
    int stream_size0, stream_size1;
    double *timer0, *timer1;
    static int wait_flag = 0;
    double epsilon = 0.2; // 0.2ms
    if(num_remote) {
        if(pure_host) {
            // do {
            //     hret = margo_wait_any(num_remote, breq, &req_idx);
            //     if(hret != HG_SUCCESS) {
            //         fprintf(stderr,
            //             "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
            //                 __func__, req_idx, hret);
            //         list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
            //                                     struct size_t_list_entry, entry) {
            //             obj_data_free_cuda(remote_device_od[req_ent->value]);
            //             list_del(&req_ent->entry);
            //             free(req_ent);
            //         }
            //         free(remote_device_od);
            //         for(int i=0; i<remote_stream_size; i++) {
            //             cudaStreamDestroy(remote_stream[i]);
            //         }
            //         free(remote_stream);
            //         for(int i=0; i<num_remote; i++) {
            //             obj_data_free(remote_od[i]);
            //         }
            //         free(remote_od);
            //         free(bhndl);
            //         free(breq);
            //         free(bin);
            //         free(remote_odsc_tab);
            //         free(return_od);
            //         if(init_pattern) {
            //             list_del(&getobj_ent->entry);
            //             free(getobj_ent->var_name);
            //             free(getobj_ent);
            //         }
            //         return dspaces_ERR_MERCURY;
            //     }
            //     // break when all req are finished
            //     if(req_idx == num_remote) {
            //         break;
            //     }
            //     breq[req_idx] = MARGO_REQUEST_NULL;
            //     margo_get_output(bhndl[req_idx], &bresp);
            //     remote_device_od[req_idx] = obj_data_alloc_cuda(&remote_odsc_tab[req_idx]);
            //     // H->D async transfer
            //     h2d_size = (qodsc.size)*bbox_volume(&remote_odsc_tab[req_idx].bb);
            //     curet = cudaMemcpyAsync(remote_device_od[req_idx]->data,
            //                             remote_od[req_idx]->data, h2d_size,
            //                             cudaMemcpyHostToDevice,
            //                             remote_stream[req_idx%remote_stream_size]);
            //     if(curet != cudaSuccess) {
            //         fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
            //                 __func__, cudaGetErrorString(curet));
            //         obj_data_free_cuda(remote_device_od[req_idx]);
            //         list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
            //                                     struct size_t_list_entry, entry) {
            //             obj_data_free_cuda(remote_device_od[req_ent->value]);
            //             list_del(&req_ent->entry);
            //             free(req_ent);
            //         }
            //         free(remote_device_od);
            //         for(int i=0; i<remote_stream_size; i++) {
            //             cudaStreamDestroy(remote_stream[i]);
            //         }
            //         free(remote_stream);
            //         for(int i=0; i<num_remote; i++) {
            //             obj_data_free(remote_od[i]);
            //         }
            //         free(remote_od);
            //         free(bhndl);
            //         free(breq);
            //         free(bin);
            //         free(remote_odsc_tab);
            //         free(return_od);
            //         if(init_pattern) {
            //             list_del(&getobj_ent->entry);
            //             free(getobj_ent->var_name);
            //             free(getobj_ent);
            //         }
            //         return dspaces_ERR_CUDA;
            //     }
            //     ret = ssd_copy_cuda_async(return_od, remote_device_od[req_idx],
            //                             &remote_stream[req_idx%remote_stream_size]);
            //     if(ret != dspaces_SUCCESS) {
            //         fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
            //             __func__, ret);
            //         obj_data_free_cuda(remote_device_od[req_idx]);
            //         list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
            //                                     struct size_t_list_entry, entry) {
            //             obj_data_free_cuda(remote_device_od[req_ent->value]);
            //             list_del(&req_ent->entry);
            //             free(req_ent);
            //         }
            //         free(remote_device_od);
            //         for(int i=0; i<remote_stream_size; i++) {
            //             cudaStreamDestroy(remote_stream[i]);
            //         }
            //         free(remote_stream);
            //         for(int i=0; i<num_remote; i++) {
            //             obj_data_free(remote_od[i]);
            //         }
            //         free(remote_od);
            //         free(bhndl);
            //         free(breq);
            //         free(bin);
            //         free(remote_odsc_tab);
            //         free(return_od);
            //         if(init_pattern) {
            //             list_del(&getobj_ent->entry);
            //             free(getobj_ent->var_name);
            //             free(getobj_ent);
            //         }
            //         return dspaces_ERR_CUDA;
            //     }
            //     req_ent = 
            //         (struct size_t_list_entry *) malloc(sizeof(struct size_t_list_entry));
            //     req_ent->value = req_idx;
            //     list_add(&req_ent->entry, &req_done_list);
            //     margo_free_output(bhndl[req_idx], &bresp);
            //     margo_bulk_free(bin[req_idx].handle);
            //     margo_destroy(bhndl[req_idx]);
            // } while(req_idx != num_remote);

            // // beyond this point, all device_od are allocated
            // list_for_each_entry_safe(req_ent, req_tmp, &req_done_list, struct size_t_list_entry, entry) {
            //     list_del(&req_ent->entry);
            //     free(req_ent);
            // }
            alloc_cnt = 0;
            for(int i=0; i<num_remote; i++) {
                remote_device_od[i] = obj_data_alloc_cuda(&remote_odsc_tab[i]);
                margo_wait(breq[i]);
                margo_get_output(bhndl[i], &bresp);
                // H->D async transfer
                h2d_size = (qodsc.size)*bbox_volume(&remote_odsc_tab[i].bb);
                curet = cudaMemcpyAsync(remote_device_od[i]->data,
                                        remote_od[i]->data, h2d_size,
                                        cudaMemcpyHostToDevice,
                                        remote_stream[i%remote_stream_size]);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    obj_data_free_cuda(remote_device_od[i]);
                    for(int j=0; j<alloc_cnt; j++) {
                        obj_data_free_cuda(remote_device_od[j]);
                    }
                    free(remote_device_od);
                    for(int j=0; j<remote_stream_size; j++) {
                        cudaStreamDestroy(remote_stream[j]);
                    }
                    free(remote_stream);
                    for(int j=0; j<num_remote; j++) {
                        obj_data_free(remote_od[j]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                ret = ssd_copy_cuda_async(return_od, remote_device_od[i],
                                        &remote_stream[i%remote_stream_size]);
                if(ret != dspaces_SUCCESS) {
                    fprintf(stderr,
                            "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                            __func__, ret);
                    obj_data_free_cuda(remote_device_od[i]);
                    for(int j=0; j<alloc_cnt; j++) {
                        obj_data_free_cuda(remote_device_od[j]);
                    }
                    free(remote_device_od);
                    for(int j=0; j<remote_stream_size; j++) {
                        cudaStreamDestroy(remote_stream[j]);
                    }
                    free(remote_stream);
                    for(int j=0; j<num_remote; j++) {
                        obj_data_free(remote_od[j]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                margo_free_output(bhndl[i], &bresp);
                margo_bulk_free(bin[i].handle);
                margo_destroy(bhndl[i]);
                alloc_cnt++;
            }

            free(bhndl);
            free(breq);
            free(bin);

            for(int i=0; i<remote_stream_size; i++) {
                curet = cudaStreamSynchronize(remote_stream[i]);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    for(int i=0; i<remote_stream_size; i++) {
                        cudaStreamDestroy(remote_stream[i]);
                    }
                    free(remote_stream);
                    for(int i=0; i<num_remote; i++) {
                        obj_data_free_cuda(remote_device_od[i]);
                        obj_data_free(remote_od[i]);
                    }
                    free(remote_device_od);
                    free(remote_od);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                cudaStreamDestroy(remote_stream[i]);
            }
            free(remote_stream);
            for(int i=0; i<num_remote; i++) {
                obj_data_free_cuda(remote_device_od[i]);
                obj_data_free(remote_od[i]);
            }
            free(remote_device_od);
            free(remote_od);
        } else if(pure_gdr) {
            // do {
            //     hret = margo_wait_any(num_remote, breq, &req_idx);
            //     if(hret != HG_SUCCESS) {
            //         fprintf(stderr,
            //             "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
            //                 __func__, req_idx, hret);
            //         list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
            //                                     struct size_t_list_entry, entry) {
            //             obj_data_free_cuda(remote_device_od[req_ent->value]);
            //             list_del(&req_ent->entry);
            //             free(req_ent);
            //         }
            //         free(remote_device_od);
            //         for(int i=0; i<remote_stream_size; i++) {
            //             cudaStreamDestroy(remote_stream[i]);
            //         }
            //         free(remote_stream);
            //         for(int i=0; i<num_remote; i++) {
            //             obj_data_free(remote_od[i]);
            //         }
            //         free(remote_od);
            //         free(bhndl);
            //         free(breq);
            //         free(bin);
            //         free(remote_odsc_tab);
            //         free(return_od);
            //         if(init_pattern) {
            //             list_del(&getobj_ent->entry);
            //             free(getobj_ent->var_name);
            //             free(getobj_ent);
            //         }
            //         return dspaces_ERR_MERCURY;
            //     }
            //     // break when all req are finished
            //     if(req_idx == num_remote) {
            //         break;
            //     }
            //     breq[req_idx] = MARGO_REQUEST_NULL;
            //     margo_get_output(bhndl[req_idx], &bresp);
            //     ret = ssd_copy_cuda_async(return_od, remote_od[req_idx], 
            //                         &remote_stream[req_idx%remote_stream_size]);
            //     if(ret != dspaces_SUCCESS) {
            //         fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
            //             __func__, ret);
            //         for(int i=0; i<remote_stream_size; i++) {
            //             cudaStreamDestroy(remote_stream[i]);
            //         }
            //         free(remote_stream);
            //         for(int i=0; i<num_remote; i++) {
            //             obj_data_free_cuda(remote_od[i]);
            //         }
            //         free(remote_od);
            //         free(bhndl);
            //         free(breq);
            //         free(bin);
            //         free(remote_odsc_tab);
            //         free(return_od);
            //         if(init_pattern) {
            //             list_del(&getobj_ent->entry);
            //             free(getobj_ent->var_name);
            //             free(getobj_ent);
            //         }
            //         return dspaces_ERR_CUDA;
            //     }
            //     margo_free_output(bhndl[req_idx], &bresp);
            //     margo_bulk_free(bin[req_idx].handle);
            //     margo_destroy(bhndl[req_idx]);
            // } while(req_idx != num_remote);

            for(int i=0; i<num_remote; i++) {
                margo_wait(breq[i]);
                margo_get_output(bhndl[i], &bresp);
                ret = ssd_copy_cuda_async(return_od, remote_od[i], 
                                    &remote_stream[i%remote_stream_size]);
                if(ret != dspaces_SUCCESS) {
                    fprintf(stderr,
                            "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                            __func__, ret);
                    for(int j=0; j<remote_stream_size; j++) {
                        cudaStreamDestroy(remote_stream[j]);
                    }
                    free(remote_stream);
                    for(int j=0; j<num_remote; j++) {
                        obj_data_free(remote_od[j]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                margo_free_output(bhndl[i], &bresp);
                margo_bulk_free(bin[i].handle);
                margo_destroy(bhndl[i]);
            }

            free(bhndl);
            free(breq);
            free(bin);

            for(int i=0; i<remote_stream_size; i++) {
                curet = cudaStreamSynchronize(remote_stream[i]);
                if(curet != cudaSuccess) {
                    fprintf(stderr,
                            "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                            __func__, cudaGetErrorString(curet));
                    for(int i=0; i<remote_stream_size; i++) {
                        cudaStreamDestroy(remote_stream[i]);
                    }
                    free(remote_stream);
                    for(int i=0; i<num_remote; i++) {
                        obj_data_free_cuda(remote_od[i]);
                    }
                    free(remote_od);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                cudaStreamDestroy(remote_stream[i]);
            }
            free(remote_stream);
            for(int i=0; i<num_remote; i++) {
                obj_data_free_cuda(remote_od[i]);
            }
            free(remote_od);
        } else {
            do {
                gettimeofday(&start, NULL);
                hret = margo_wait_any(num_rpcs, breq, &req_idx);
                if(hret != HG_SUCCESS) {
                    fprintf(stderr,
                        "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
                            __func__, req_idx, hret);
                    list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
                                                struct size_t_list_entry, entry) {
                        obj_data_free_cuda(remote_device_od[req_ent->value]);
                        list_del(&req_ent->entry);
                        free(req_ent);
                    }
                    free(remote_device_od);
                    for(int i=0; i<host_stream_size; i++) {
                        cudaStreamDestroy(host_stream[i]);
                    }
                    free(host_stream);
                    for(int i=0; i<gdr_stream_size; i++) {
                        cudaStreamDestroy(gdr_stream[i]);
                    }
                    free(gdr_stream);
                    for(int i=0; i<num_remote; i++) {
                        obj_data_free(remote_od[i]);
                        obj_data_free_cuda(remote_od[i+num_remote]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_MERCURY;
                }
                // break when all req are finished
                if(req_idx == num_rpcs) {
                    break;
                }
                breq[req_idx] = MARGO_REQUEST_NULL;
                margo_get_output(bhndl[req_idx], &bresp);
                if(req_idx < num_odscs) { // host-based path
                    remote_device_od[req_idx] = obj_data_alloc_cuda(&remote_odsc_tab[req_idx]);
                    // H->D async transfer
                    h2d_size = (qodsc.size)*bbox_volume(&remote_odsc_tab[req_idx].bb);
                    curet = cudaMemcpyAsync(remote_device_od[req_idx]->data,
                                            remote_od[req_idx]->data, h2d_size,
                                            cudaMemcpyHostToDevice,
                                            host_stream[req_idx%host_stream_size]);
                    if(curet != cudaSuccess) {
                        fprintf(stderr, "ERROR: (%s): cudaMemcpyAsync() failed, Err Code: (%s)\n",
                                __func__, cudaGetErrorString(curet));
                        obj_data_free_cuda(remote_device_od[req_idx]);
                        list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
                                                    struct size_t_list_entry, entry) {
                            obj_data_free_cuda(remote_device_od[req_ent->value]);
                            list_del(&req_ent->entry);
                            free(req_ent);
                        }
                        free(remote_device_od);
                        for(int i=0; i<host_stream_size; i++) {
                            cudaStreamDestroy(host_stream[i]);
                        }
                        free(host_stream);
                        for(int i=0; i<gdr_stream_size; i++) {
                            cudaStreamDestroy(gdr_stream[i]);
                        }
                        free(gdr_stream);
                        for(int j = 0; j < num_remote; j++) {
                            obj_data_free(remote_od[j]);
                            obj_data_free_cuda(remote_od[j+num_remote]);
                        }
                        free(remote_od);
                        free(bhndl);
                        free(breq);
                        free(bin);
                        free(remote_odsc_tab);
                        free(return_od);
                        if(init_pattern) {
                            list_del(&getobj_ent->entry);
                            free(getobj_ent->var_name);
                            free(getobj_ent);
                        }
                        return dspaces_ERR_CUDA;
                    }
                    // track allocated device_od
                    req_ent = 
                        (struct size_t_list_entry *) malloc(sizeof(struct size_t_list_entry));
                    req_ent->value = req_idx;
                    list_add(&req_ent->entry, &req_done_list);
                    ret = ssd_copy_cuda_async(return_od, remote_device_od[req_idx],
                                        &host_stream[req_idx%host_stream_size]);
                    gettimeofday(&end, NULL);
                    host_timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                } else { // GDR path
                    ret = ssd_copy_cuda_async(return_od, remote_od[req_idx], 
                                    &gdr_stream[(req_idx-num_remote)%gdr_stream_size]);
                    gettimeofday(&end, NULL);
                    gdr_timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                }
                if(ret != dspaces_SUCCESS) {
                    fprintf(stderr, "ERROR: (%s): ssd_copy_cuda_async() failed, Err Code: (%d)\n",
                        __func__, ret);
                    list_for_each_entry_safe(req_ent, req_tmp, &req_done_list,
                                                struct size_t_list_entry, entry) {
                        obj_data_free_cuda(remote_device_od[req_ent->value]);
                        list_del(&req_ent->entry);
                        free(req_ent);
                    }
                    free(remote_device_od);
                    for(int i=0; i<host_stream_size; i++) {
                        cudaStreamDestroy(host_stream[i]);
                    }
                    free(host_stream);
                    for(int i=0; i<gdr_stream_size; i++) {
                        cudaStreamDestroy(gdr_stream[i]);
                    }
                    free(gdr_stream);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                    }
                    free(remote_od);
                    free(bhndl);
                    free(breq);
                    free(bin);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                margo_free_output(bhndl[req_idx], &bresp);
                margo_bulk_free(bin[req_idx].handle);
                margo_destroy(bhndl[req_idx]);
            } while(req_idx != num_rpcs);

            // beyond this point, all device_od are allocated
            list_for_each_entry_safe(req_ent, req_tmp, &req_done_list, struct size_t_list_entry, entry) {
                list_del(&req_ent->entry);
                free(req_ent);
            }

            free(bhndl);
            free(breq);
            free(bin);

            if(wait_flag == 0) {
                stream0 = gdr_stream;
                stream_size0 = gdr_stream_size;
                timer0 = &gdr_timer;
                stream1 = host_stream;
                stream_size1 = host_stream_size;
                timer1 = &host_timer;
            } else {
                stream0 = host_stream;
                stream_size0 = host_stream_size;
                timer0 = &host_timer;
                stream1 = gdr_stream;
                stream_size1 = gdr_stream_size;
                timer1 = &gdr_timer;
            }

            for(int i=0; i<stream_size0; i++) {
                gettimeofday(&start, NULL);
                curet = cudaStreamSynchronize(stream0[i]);
                gettimeofday(&end, NULL);
                *timer0 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                if(curet != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                                __func__, cudaGetErrorString(curet));
                    for(int j=i; j<stream_size0; j++) {
                        cudaStreamDestroy(stream0[j]);
                    }
                    free(stream0);
                    for(int j=0; j<stream_size1; j++) {
                        cudaStreamDestroy(stream1[j]);
                    }
                    free(stream1);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                        obj_data_free_cuda(remote_device_od[j]);
                    }
                    free(remote_device_od);
                    free(remote_od);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                cudaStreamDestroy(stream0[i]);
            }

            for(int i=0; i<stream_size1; i++) {
                gettimeofday(&start, NULL);
                curet = cudaStreamSynchronize(stream1[i]);
                gettimeofday(&end, NULL);
                *timer1 += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                if(curet != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cudaStreamSynchronize() failed, Err Code: (%s)\n",
                                __func__, cudaGetErrorString(curet));
                    free(stream0);
                    for(int j=i; j<stream_size1; j++) {
                        cudaStreamDestroy(stream1[j]);
                    }
                    free(stream1);
                    for(int j = 0; j < num_remote; j++) {
                        obj_data_free(remote_od[j]);
                        obj_data_free_cuda(remote_od[j+num_remote]);
                        obj_data_free_cuda(remote_device_od[j]);
                    }
                    free(remote_device_od);
                    free(remote_od);
                    free(remote_odsc_tab);
                    free(return_od);
                    if(init_pattern) {
                        list_del(&getobj_ent->entry);
                        free(getobj_ent->var_name);
                        free(getobj_ent);
                    }
                    return dspaces_ERR_CUDA;
                }
                cudaStreamDestroy(stream1[i]);
            }

            free(gdr_stream);
            free(host_stream);

            if(*timer1 > stream_size1*epsilon) {
                // 2nd request takes longer time, tune ratio
                if(gdr_timer < host_timer) {
                    if(host_timer - gdr_timer > 1e-3) {
                        gdr_ratio += ((host_timer - gdr_timer) / host_timer) * (1-gdr_ratio);
                        host_ratio = 1 - gdr_ratio;
                    }
                } else {
                    if(gdr_timer - host_timer > 1e-3) {
                        gdr_ratio -= ((gdr_timer - host_timer) / gdr_timer) * (gdr_ratio-0);
                        host_ratio = 1 - gdr_ratio;
                    }
                }
            } else {
                // 2nd request finishes no later than the 1st request
                // swap request by setting flag = 1
                wait_flag == 0 ? 1:0;
            }

            DEBUG_OUT("ts = %u, gdr_ratio = %lf, host_ratio = %lf,"
                        "gdr_time = %lf, host_time = %lf\n", qodsc.version, gdr_ratio, host_ratio, 
                            gdr_timer, host_timer);
            
            for(int j = 0; j < num_remote; j++) {
                obj_data_free(remote_od[j]);
                obj_data_free_cuda(remote_od[j+num_remote]);
                obj_data_free_cuda(remote_device_od[j]);
            }
            free(remote_device_od);
            free(remote_od);
        }
    }
    free(remote_odsc_tab);
    free(return_od);
    
    return ret;
}

int dspaces_put_meta(dspaces_client_t client, const char *name, int version,
                     const void *data, unsigned int len)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_size_t rdma_length = len;
    hg_return_t hret;
    put_meta_in_t in;
    bulk_out_t out;

    int ret = dspaces_SUCCESS;

    DEBUG_OUT("posting metadata for `%s`, version %d with length %i bytes.\n",
              name, version, len);

    in.name = strdup(name);
    in.length = len;
    in.version = version;
    hret = margo_bulk_create(client->mid, 1, (void **)&data, &rdma_length,
                             HG_BULK_READ_ONLY, &in.handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failed\n", __func__);
        return dspaces_ERR_MERCURY;
    }

    get_meta_server_address(client, &server_addr);
    hret = margo_create(client->mid, server_addr, client->put_meta_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        margo_bulk_free(in.handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed\n", __func__);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_get_output() failed\n", __func__);
        margo_bulk_free(in.handle);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    DEBUG_OUT("metadata posted successfully.\n");

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_bulk_free(in.handle);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);

    return (ret);
}

int dspaces_put_local(dspaces_client_t client, const char *var_name,
                      unsigned int ver, int elem_size, int ndim, uint64_t *lb,
                      uint64_t *ub, void *data)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    int ret = dspaces_SUCCESS;

    if(client->listener_init == 0) {
        ret = dspaces_init_listener(client);
        if(ret != dspaces_SUCCESS) {
            return (ret);
        }
    }

    client->local_put_count++;

    obj_descriptor odsc = {.version = ver,
                           .st = st,
                           .flags = DS_CLIENT_STORAGE,
                           .size = elem_size,
                           .bb = {
                               .num_dims = ndim,
                           }};

    hg_addr_t owner_addr;
    size_t owner_addr_size = 128;

    margo_addr_self(client->mid, &owner_addr);
    margo_addr_to_string(client->mid, odsc.owner, &owner_addr_size, owner_addr);
    margo_addr_free(client->mid, owner_addr);

    memset(odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);

    memcpy(odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);

    strncpy(odsc.name, var_name, sizeof(odsc.name) - 1);
    odsc.name[sizeof(odsc.name) - 1] = '\0';

    odsc_gdim_t in;
    bulk_out_t out;
    struct obj_data *od;
    od = obj_data_alloc_with_data(&odsc, data);

    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &od->gdim);

    ABT_mutex_lock(client->ls_mutex);
    ls_add_obj(client->dcg->ls, od);
    DEBUG_OUT("Added into local_storage\n");
    ABT_mutex_unlock(client->ls_mutex);

    in.odsc_gdim.size = sizeof(odsc);
    in.odsc_gdim.raw_odsc = (char *)(&odsc);
    in.odsc_gdim.gdim_size = sizeof(struct global_dimension);
    in.odsc_gdim.raw_gdim = (char *)(&od->gdim);

    DEBUG_OUT("sending object information %s \n", obj_desc_sprint(&odsc));

    get_server_address(client, &server_addr);
    /* create handle */
    hret =
        margo_create(client->mid, server_addr, client->put_local_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_forward() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s):  margo_get_output() failed\n", __func__);
        margo_destroy(handle);
        return dspaces_ERR_MERCURY;
    }

    ret = out.ret;
    margo_free_output(handle, &out);
    margo_destroy(handle);
    margo_addr_free(client->mid, server_addr);

    return ret;
}

static int get_odscs(dspaces_client_t client, obj_descriptor *odsc, int timeout,
                     obj_descriptor **odsc_tab)
{
    struct global_dimension od_gdim;
    int num_odscs;
    hg_addr_t server_addr;
    hg_return_t hret;
    hg_handle_t handle;

    odsc_gdim_t in;
    odsc_list_t out;

    in.odsc_gdim.size = sizeof(*odsc);
    in.odsc_gdim.raw_odsc = (char *)odsc;
    in.param = timeout;

    set_global_dimension(&(client->dcg->gdim_list), odsc->name,
                         &(client->dcg->default_gdim), &od_gdim);
    in.odsc_gdim.gdim_size = sizeof(od_gdim);
    in.odsc_gdim.raw_gdim = (char *)(&od_gdim);

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->query_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_create() failed with %d.\n", __func__,
                hret);
        return (0);
    }
    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_forward() failed with %d.\n",
                __func__, hret);
        margo_destroy(handle);
        return (0);
    }
    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_get_output() failed with %d.\n",
                __func__, hret);
        margo_destroy(handle);
        return (0);
    }

    num_odscs = (out.odsc_list.size) / sizeof(obj_descriptor);
    *odsc_tab = malloc(out.odsc_list.size);
    memcpy(*odsc_tab, out.odsc_list.raw_odsc, out.odsc_list.size);
    margo_free_output(handle, &out);
    margo_addr_free(client->mid, server_addr);
    margo_destroy(handle);

    return (num_odscs);
}

int dspaces_aget(dspaces_client_t client, const char *var_name,
                 unsigned int ver, int ndim, uint64_t *lb, uint64_t *ub,
                 void **data, int timeout)
{
    obj_descriptor odsc;
    obj_descriptor *odsc_tab;
    int num_odscs;
    int elem_size;
    int num_elem = 1;
    int i;
    int ret = dspaces_SUCCESS;

    fill_odsc(var_name, ver, 0, ndim, lb, ub, &odsc);

    num_odscs = get_odscs(client, &odsc, timeout, &odsc_tab);

    DEBUG_OUT("Finished query - need to fetch %d objects\n", num_odscs);
    for(int i = 0; i < num_odscs; ++i) {
        DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tab[i]));
    }

    // send request to get the obj_desc
    if(num_odscs != 0)
        elem_size = odsc_tab[0].size;
    odsc.size = elem_size;
    for(i = 0; i < ndim; i++) {
        num_elem *= (ub[i] - lb[i]) + 1;
    }
    DEBUG_OUT("data buffer size is %d\n", num_elem * elem_size);
    *data = malloc(num_elem * elem_size);
    get_data(client, num_odscs, odsc, odsc_tab, *data);

    return ret;
}

int dspaces_cpu_get(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub, void *data,
                int timeout)
{
    obj_descriptor odsc;
    obj_descriptor *odsc_tab;
    int num_odscs;
    int ret = dspaces_SUCCESS;

    fill_odsc(var_name, ver, elem_size, ndim, lb, ub, &odsc);

    DEBUG_OUT("Querying %s with timeout %d\n", obj_desc_sprint(&odsc), timeout);

    num_odscs = get_odscs(client, &odsc, timeout, &odsc_tab);

    DEBUG_OUT("Finished query - need to fetch %d objects\n", num_odscs);
    for(int i = 0; i < num_odscs; ++i) {
        DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tab[i]));
    }

    // send request to get the obj_desc
    if(num_odscs != 0) {
        get_data(client, num_odscs, odsc, odsc_tab, data);
        free(odsc_tab);
    }

    return (ret);
}

int dspaces_cuda_get(dspaces_client_t client, const char *var_name, unsigned int ver,
                     int elem_size, int ndim, uint64_t *lb, uint64_t *ub, void *data,
                     int timeout, double* ttime, double* ctime)
{
    
    struct timeval start, end;
    double timer = 0; // timer in second
    obj_descriptor odsc;
    obj_descriptor *odsc_tab;
    int num_odscs;
    int ret = dspaces_SUCCESS;
    int curet;

    if(client->cuda_info.cuda_get_mode ==6) { // Dual-Channel-Dual-Staging
        gettimeofday(&start, NULL);
        ret = dspaces_cuda_dcds_get(client, var_name, ver, elem_size, ndim, lb, ub, data,
                                timeout, ttime, ctime);
        gettimeofday(&end, NULL);
        timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
    } else {

        fill_odsc(var_name, ver, elem_size, ndim, lb, ub, &odsc);

        DEBUG_OUT("Querying %s with timeout %d\n", obj_desc_sprint(&odsc), timeout);

        num_odscs = get_odscs(client, &odsc, timeout, &odsc_tab);

        DEBUG_OUT("Finished query - need to fetch %d objects\n", num_odscs);
        for(int i = 0; i < num_odscs; ++i) {
            DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tab[i]));
        }


        // send request to get the obj_desc
        if(num_odscs != 0) {
            switch (client->cuda_info.cuda_get_mode)
            {
            // Baseline
            case 1:
            {
                size_t rdma_size = elem_size*bbox_volume(&odsc.bb);
                void* buffer = (void*) malloc(rdma_size);
                gettimeofday(&start, NULL);
                get_data_baseline(client, num_odscs, odsc, odsc_tab, buffer, ctime);
                curet = cudaMemcpy(data, buffer, rdma_size, cudaMemcpyHostToDevice);
                if(curet != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cudaMemcpy() failed, Err Code: (%s)\n",
                             __func__, cudaGetErrorString(curet));
                    ret = dspaces_ERR_CUDA;
                }
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                free(buffer);
                break;
            }
            // GDR
            case 2:
            {
                gettimeofday(&start, NULL);
                ret = get_data_gdr(client, num_odscs, odsc, odsc_tab, data, ctime);
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                break;
            }
            // Hybrid
            case 3:
            {
                gettimeofday(&start, NULL);
                ret = get_data_hybrid(client, num_odscs, odsc, odsc_tab, data, ctime);
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                break;
            }
            // Heuristic
            case 4:
            {
                gettimeofday(&start, NULL);
                ret = get_data_heuristic(client, num_odscs, odsc, odsc_tab, data, ctime);
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                break;
            }
            // Dual-Channel
            case 5:
            {
                gettimeofday(&start, NULL);
                ret = get_data_dual_channel_v2(client, num_odscs, odsc, odsc_tab, data, ctime);
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                break;
            }
            default:
            {
                size_t rdma_size = elem_size*bbox_volume(&odsc.bb);
                void* buffer = (void*) malloc(rdma_size);
                gettimeofday(&start, NULL);
                get_data_baseline(client, num_odscs, odsc, odsc_tab, buffer, ctime);
                curet = cudaMemcpy(data, buffer, rdma_size, cudaMemcpyHostToDevice);
                if(curet != cudaSuccess) {
                    fprintf(stderr, "ERROR: (%s): cudaMemcpy() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(curet));
                    ret = dspaces_ERR_CUDA;
                }
                gettimeofday(&end, NULL);
                timer += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3;
                free(buffer);
                break;
            }
            }
            
            free(odsc_tab);
        }
    }

    *ttime = timer - *ctime;
    return (ret);
}

int dspaces_get(dspaces_client_t client, const char *var_name, unsigned int ver,
                int elem_size, int ndim, uint64_t *lb, uint64_t *ub, void *data,
                int timeout)
{
    int ret;
    double ttime, ctime;
    struct cudaPointerAttributes ptr_attr;
    CUDA_ASSERTRT(cudaPointerGetAttributes(&ptr_attr, data));
    if(ptr_attr.type == cudaMemoryTypeDevice) {
        ret = dspaces_cuda_get(client, var_name, ver, elem_size, ndim, lb, ub, data, timeout, &ttime, &ctime);
    } else {
        ret = dspaces_cpu_get(client, var_name, ver, elem_size, ndim, lb, ub, data, timeout);
    }
    return ret;
}

int dspaces_get_meta(dspaces_client_t client, const char *name, int mode,
                     int current, int *version, void **data, unsigned int *len)
{
    query_meta_in_t in;
    query_meta_out_t out;
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_bulk_t bulk_handle;
    hg_return_t hret;

    in.name = strdup(name);
    in.version = current;
    in.mode = mode;

    DEBUG_OUT("querying meta data '%s' version %d (mode %d).\n", name, current,
              mode);

    get_meta_server_address(client, &server_addr);
    hret =
        margo_create(client->mid, server_addr, client->query_meta_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_create() failed with %d.\n", __func__,
                hret);
        goto err_hg;
    }
    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_forward() failed with %d.\n",
                __func__, hret);
        goto err_hg_handle;
    }
    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_get_output() failed with %d.\n",
                __func__, hret);
        goto err_hg_output;
    }

    DEBUG_OUT("Replied with version %d.\n", out.version);

    if(out.size) {
        DEBUG_OUT("fetching %zi bytes.\n", out.size);
        *data = malloc(out.size);
        hret = margo_bulk_create(client->mid, 1, data, &out.size,
                                 HG_BULK_WRITE_ONLY, &bulk_handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: %s: margo_bulk_create() failed with %d.\n",
                    __func__, hret);
            goto err_free;
        }
        hret = margo_bulk_transfer(client->mid, HG_BULK_PULL, server_addr,
                                   out.handle, 0, bulk_handle, 0, out.size);
        if(hret != HG_SUCCESS) {
            fprintf(stderr,
                    "ERROR: %s: margo_bulk_transfer() failed with %d.\n",
                    __func__, hret);
            goto err_bulk;
        }
        DEBUG_OUT("metadata for '%s', version %d retrieved successfully.\n",
                  name, out.version);
    } else {
        DEBUG_OUT("Metadata is empty.\n");
        *data = NULL;
    }

    *len = out.size;
    *version = out.version;

    margo_bulk_free(bulk_handle);
    margo_free_output(handle, &out);
    margo_destroy(handle);

    return dspaces_SUCCESS;

err_bulk:
    margo_bulk_free(bulk_handle);
err_free:
    free(*data);
err_hg_output:
    margo_free_output(handle, &out);
err_hg_handle:
    margo_destroy(handle);
err_hg:
    free(in.name);
    return dspaces_ERR_MERCURY;
}

static void get_local_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    bulk_in_t in;
    bulk_out_t out;
    hg_bulk_t bulk_handle;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("Received rpc to get data\n");

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s. margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));

    DEBUG_OUT("%s\n", obj_desc_sprint(&in_odsc));

    ABT_mutex_lock(client->putlocal_subdrain_mutex);

    struct subdrain_list_entry *e =
        lookup_putlocal_subdrain_list(&client->dcg->putlocal_subdrain_list, in_odsc);
    if(e) {
        e->get_count++;
    }

    ABT_mutex_unlock(client->putlocal_subdrain_mutex);

    struct obj_data *od, *from_obj;

    ABT_mutex_lock(client->ls_mutex);
    from_obj = ls_find(client->dcg->ls, &in_odsc);
    ABT_mutex_unlock(client->ls_mutex);
    if(!from_obj) {
        fprintf(stderr,
                "DATASPACES: WARNING handling %s: Object not found in local "
                "storage\n",
                __func__);
        if(e) {
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            e->get_count--;
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        }
    }

    od = obj_data_alloc(&in_odsc);
    if(!od) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: object allocation failed\n",
                __func__);
        if(e) {
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            e->get_count--;
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        }
    }

    if(from_obj->data == NULL) {
        fprintf(
            stderr,
            "DATASPACES: ERROR handling %s: object data allocation failed\n",
            __func__);
        if(e) {
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            e->get_count--;
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        }
    }

    ssd_copy(od, from_obj);
    DEBUG_OUT("After ssd_copy\n");

    hg_size_t size = (in_odsc.size) * bbox_volume(&(in_odsc.bb));
    void *buffer = (void *)od->data;

    hret = margo_bulk_create(mid, 1, (void **)&buffer, &size, HG_BULK_READ_ONLY,
                             &bulk_handle);

    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_bulk_create() failed\n",
                __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        if(e) {
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            e->get_count--;
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        }
        return;
    }

    hret = margo_bulk_transfer(mid, HG_BULK_PUSH, info->addr, in.handle, 0,
                               bulk_handle, 0, size);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_bulk_transfer() failed "
                "(%d)\n",
                __func__, hret);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_bulk_free(bulk_handle);
        margo_destroy(handle);
        if(e) {
            ABT_mutex_lock(client->putlocal_subdrain_mutex);
            e->get_count--;
            ABT_mutex_unlock(client->putlocal_subdrain_mutex);
        }
        return;
    }
    margo_bulk_free(bulk_handle);
    if(e) {
        ABT_mutex_lock(client->putlocal_subdrain_mutex);
        e->get_count--;
        if(e->get_count == 0) {
            ABT_cond_signal(e->delete_cond);
        }
        ABT_mutex_unlock(client->putlocal_subdrain_mutex);
    }
    out.ret = dspaces_SUCCESS;
    obj_data_free(od);
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(get_local_rpc)

static void drain_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    bulk_in_t in;
    bulk_out_t out;
    hg_bulk_t bulk_handle;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("Received rpc to drain data\n");

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));

    DEBUG_OUT("%s\n", obj_desc_sprint(&in_odsc));

    struct obj_data *from_obj;

    ABT_mutex_lock(client->ls_mutex);
    from_obj = ls_find(client->dcg->ls, &in_odsc);
    ABT_mutex_unlock(client->ls_mutex);
    if(!from_obj) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s:"
                "Object not found in client's local storage.\n Make sure MAX "
                "version is set appropriately in dataspaces.conf\n",
                __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        return;
    }

    hg_size_t size = (in_odsc.size) * bbox_volume(&(in_odsc.bb));
    void *buffer = (void *)from_obj->data;

    hret = margo_bulk_create(mid, 1, (void **)&buffer, &size, HG_BULK_READ_ONLY,
                             &bulk_handle);

    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_bulk_create() failed\n",
                __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        return;
    }

    hret = margo_bulk_transfer(mid, HG_BULK_PUSH, info->addr, in.handle, 0,
                               bulk_handle, 0, size);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_bulk_transfer() failed\n",
                __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_bulk_free(bulk_handle);
        margo_destroy(handle);
        return;
    }
    margo_bulk_free(bulk_handle);

    out.ret = dspaces_SUCCESS;
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
    // delete object from local storage
    DEBUG_OUT("Finished draining %s\n", obj_desc_sprint(&from_obj->obj_desc));
    ABT_mutex_lock(client->ls_mutex);
    ls_try_remove_free(client->dcg->ls, from_obj);
    ABT_mutex_unlock(client->ls_mutex);

    ABT_mutex_lock(client->drain_mutex);
    client->local_put_count--;
    if(client->local_put_count == 0 && client->f_final) {
        DEBUG_OUT("signaling all objects drained.\n");
        ABT_cond_signal(client->drain_cond);
    }
    ABT_mutex_unlock(client->drain_mutex);
    DEBUG_OUT("%d objects left to drain...\n", client->local_put_count);
}
DEFINE_MARGO_RPC_HANDLER(drain_rpc)

static struct dspaces_sub_handle *dspaces_get_sub(dspaces_client_t client,
                                                  int sub_id)
{
    int listidx = sub_id % SUB_HASH_SIZE;
    struct sub_list_node *node;

    node = client->sub_lists[listidx];
    while(node) {
        if(node->id == sub_id) {
            return (node->subh);
        }
    }

    fprintf(stderr,
            "WARNING: received notification for unknown subscription id %d. "
            "This shouldn't happen.\n",
            sub_id);
    return (NULL);
}

static void dspaces_move_sub(dspaces_client_t client, int sub_id)
{
    int listidx = sub_id % SUB_HASH_SIZE;
    struct sub_list_node *node, **nodep;

    nodep = &client->sub_lists[listidx];
    while(*nodep && (*nodep)->id != sub_id) {
        nodep = &((*nodep)->next);
    }

    if(!*nodep) {
        fprintf(stderr,
                "WARNING: trying to mark unknown sub %d done. This shouldn't "
                "happen.\n",
                sub_id);
        return;
    }

    node = *nodep;
    *nodep = node->next;
    node->next = client->done_list;
    client->done_list = node;
}

static void free_sub_req(struct dspaces_req *req)
{
    if(!req) {
        return;
    }

    free(req->var_name);
    free(req->lb);
    free(req->ub);
    free(req);
}

static void notify_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);
    odsc_list_t in;
    struct dspaces_sub_handle *subh;
    int sub_id;
    int num_odscs;
    obj_descriptor *odsc_tab;
    void *data;
    size_t data_size;
    int i;

    margo_get_input(handle, &in);
    sub_id = in.param;

    DEBUG_OUT("Received notification for sub %d\n", sub_id);
    ABT_mutex_lock(client->sub_mutex);
    subh = dspaces_get_sub(client, sub_id);
    if(subh->status == DSPACES_SUB_WAIT) {
        ABT_mutex_unlock(client->sub_mutex);

        num_odscs = (in.odsc_list.size) / sizeof(obj_descriptor);
        odsc_tab = malloc(in.odsc_list.size);
        memcpy(odsc_tab, in.odsc_list.raw_odsc, in.odsc_list.size);

        DEBUG_OUT("Satisfying subscription requires fetching %d objects:\n",
                  num_odscs);
        for(i = 0; i < num_odscs; i++) {
            DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tab[i]));
        }

        data_size = subh->q_odsc.size;
        for(i = 0; i < subh->q_odsc.bb.num_dims; i++) {
            data_size *=
                (subh->q_odsc.bb.ub.c[i] - subh->q_odsc.bb.lb.c[i]) + 1;
        }
        data = malloc(data_size);

        if(num_odscs) {
            get_data(client, num_odscs, subh->q_odsc, odsc_tab, data);
        }
    } else {
        fprintf(stderr,
                "WARNING: got notification, but sub status was not "
                "DSPACES_SUB_WAIT (%i)\n",
                subh->status);
        ABT_mutex_unlock(client->sub_mutex);
        odsc_tab = NULL;
        data = NULL;
    }

    margo_free_input(handle, &in);
    margo_destroy(handle);

    ABT_mutex_lock(client->sub_mutex);
    if(subh->status == DSPACES_SUB_WAIT) {
        subh->req->buf = data;
        subh->status = DSPACES_SUB_RUNNING;
    } else if(data) {
        // subscription was cancelled
        free(data);
        data = NULL;
    }
    ABT_mutex_unlock(client->sub_mutex);

    if(data) {
        subh->result = subh->cb(client, subh->req, subh->arg);
    }

    ABT_mutex_lock(client->sub_mutex);
    client->pending_sub--;
    dspaces_move_sub(client, sub_id);
    subh->status = DSPACES_SUB_DONE;
    ABT_cond_signal(client->sub_cond);
    ABT_mutex_unlock(client->sub_mutex);

    if(odsc_tab) {
        free(odsc_tab);
    }
    free_sub_req(subh->req);
}
DEFINE_MARGO_RPC_HANDLER(notify_rpc)

static void notify_ods_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);

    hg_return_t hret;
    odsc_list_t qin;
    struct subods_list_entry *subod_ent;
    int sub_ods_id;
    int num_odscs;
    obj_descriptor *odsc_tab;

    bulk_in_t *bin;
    struct obj_data **od;
    hg_handle_t *bhndl;
    margo_request *breq;
    size_t req_idx;
    bulk_out_t bresp;
    

    margo_get_input(handle, &qin);
    sub_ods_id = qin.param;

    DEBUG_OUT("Received notification for sub %d\n", sub_ods_id);
    ABT_mutex_lock(client->sub_ods_mutex);
    subod_ent = lookup_subods_list(&client->dcg->sub_ods_list, sub_ods_id);
    if(!subod_ent) {
        ABT_mutex_unlock(client->sub_ods_mutex);
        fprintf(stderr,
            "WARNING: received notification for unknown ods subscription id %d. "
            "This shouldn't happen.\n",
            sub_ods_id);
        margo_free_input(handle, &qin);
        margo_destroy(handle);
        return ;
    }
    if(subod_ent->status == DSPACES_SUB_WAIT) {
        ABT_mutex_unlock(client->sub_ods_mutex);

        num_odscs = (qin.odsc_list.size) / sizeof(obj_descriptor);
        odsc_tab = (obj_descriptor*) malloc(qin.odsc_list.size);
        memcpy(odsc_tab, qin.odsc_list.raw_odsc, qin.odsc_list.size);

        DEBUG_OUT("Satisfying subscription requires fetching %d objects:\n",
                  num_odscs);
        for(int i = 0; i < num_odscs; i++) {
            DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tab[i]));
        }

        // TODO: add memory check
        if(num_odscs) {
            bin = (bulk_in_t*) malloc(sizeof(bulk_in_t) * num_odscs);
            od = (struct obj_data**) malloc(num_odscs * sizeof(struct obj_data*));
            bhndl = (hg_handle_t*) malloc(sizeof(hg_handle_t) * num_odscs);
            breq = (margo_request*) malloc(sizeof(margo_request) * num_odscs);

            for(int i=0; i<num_odscs; i++) {
                od[i] = obj_data_alloc(&odsc_tab[i]);
                bin[i].odsc.size = sizeof(obj_descriptor);
                bin[i].odsc.raw_odsc = (char *)(&odsc_tab[i]);
                hg_size_t rdma_size = (subod_ent->qodsc.size) * bbox_volume(&odsc_tab[i].bb);
                margo_bulk_create(client->mid, 1, (void **)(&(od[i]->data)), &rdma_size,
                                HG_BULK_WRITE_ONLY, &bin[i].handle);
                hg_addr_t bserver_addr;
                margo_addr_lookup(client->mid, odsc_tab[i].owner, &bserver_addr);
                if(odsc_tab[i].flags & DS_CLIENT_STORAGE) {
                    DEBUG_OUT("retrieving object from client-local storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_local_id,
                                &bhndl[i]);
                } else {
                    DEBUG_OUT("retrieving object from server storage.\n");
                    margo_create(client->mid, bserver_addr, client->get_id, &bhndl[i]);
                }
                margo_iforward(bhndl[i], &bin[i], &breq[i]);
                margo_addr_free(client->mid, bserver_addr);
            }

            do {
                hret = margo_wait_any(num_odscs, breq, &req_idx);
                if(hret != HG_SUCCESS) {
                    fprintf(stderr,
                        "ERROR: (%s): margo_wait_any() failed, idx = %zu. Err Code = %d.\n",
                        __func__, req_idx, hret);
                    for(int i=0; i<num_odscs; i++) {
                        margo_bulk_free(bin[i].handle);
                        margo_destroy(bhndl[i]);
                        obj_data_free(od[i]);
                    }
                    free(breq);
                    free(bhndl);
                    free(od);
                    free(bin);
                    free(odsc_tab);
                    margo_free_input(handle, &qin);
                    margo_destroy(handle);
                    return ; 
                }
                // break when all req are finished
                if(req_idx == num_odscs) {
                    break;
                }
                breq[req_idx] = MARGO_REQUEST_NULL;
                margo_get_output(bhndl[req_idx], &bresp);
                margo_free_output(bhndl[req_idx], &bresp);
                margo_bulk_free(bin[req_idx].handle);
                margo_destroy(bhndl[req_idx]);
                // add od to local storage
                ABT_mutex_lock(client->ls_mutex);
                ls_add_obj(client->dcg->ls, od[req_idx]);
                ABT_mutex_unlock(client->ls_mutex);

            } while (req_idx != num_odscs);

            free(breq);
            free(bhndl);
            free(od);
            free(bin);

        }

        free(odsc_tab);

    } else {
        ABT_mutex_unlock(client->sub_ods_mutex);
        fprintf(stderr,
                "WARNING: got ods notification, but sub status was not "
                "DSPACES_SUB_WAIT (%i)\n",
                subod_ent->status);
        
        margo_free_input(handle, &qin);
        margo_destroy(handle);
        return ;
    }

    margo_free_input(handle, &qin);
    margo_destroy(handle);

    ABT_mutex_lock(client->sub_ods_mutex);
    list_del(&subod_ent->entry);
    free(subod_ent);
    ABT_mutex_unlock(client->sub_ods_mutex);

}
DEFINE_MARGO_RPC_HANDLER(notify_ods_rpc)

static void register_client_sub(dspaces_client_t client,
                                struct dspaces_sub_handle *subh)
{
    int listidx = subh->id % SUB_HASH_SIZE;
    struct sub_list_node **node = &client->sub_lists[listidx];

    while(*node) {
        node = &((*node)->next);
    }

    *node = malloc(sizeof(**node));
    (*node)->next = NULL;
    (*node)->subh = subh;
    (*node)->id = subh->id;
}

struct dspaces_sub_handle *dspaces_sub(dspaces_client_t client,
                                       const char *var_name, unsigned int ver,
                                       int elem_size, int ndim, uint64_t *lb,
                                       uint64_t *ub, dspaces_sub_fn sub_cb,
                                       void *arg)
{
    hg_addr_t my_addr, server_addr;
    hg_handle_t handle;
    margo_request req;
    hg_return_t hret;
    struct dspaces_sub_handle *subh;
    odsc_gdim_t in;
    struct global_dimension od_gdim;
    size_t owner_addr_size = 128;
    int ret;

    if(client->listener_init == 0) {
        ret = dspaces_init_listener(client);
        if(ret != dspaces_SUCCESS) {
            return (DSPACES_SUB_FAIL);
        }
    }

    subh = malloc(sizeof(*subh));

    subh->req = malloc(sizeof(*subh->req));
    subh->req->var_name = strdup(var_name);
    subh->req->ver = ver;
    subh->req->elem_size = elem_size;
    subh->req->ndim = ndim;
    subh->req->lb = malloc(sizeof(*subh->req->lb) * ndim);
    subh->req->ub = malloc(sizeof(*subh->req->ub) * ndim);
    memcpy(subh->req->lb, lb, ndim * sizeof(*lb));
    memcpy(subh->req->ub, ub, ndim * sizeof(*ub));

    subh->q_odsc.version = ver;
    subh->q_odsc.st = st;
    subh->q_odsc.size = elem_size;
    subh->q_odsc.bb.num_dims = ndim;

    subh->arg = arg;
    subh->cb = sub_cb;

    ABT_mutex_lock(client->sub_mutex);
    client->pending_sub++;
    subh->id = client->sub_serial++;
    register_client_sub(client, subh);
    subh->status = DSPACES_SUB_WAIT;
    ABT_mutex_unlock(client->sub_mutex);

    memset(subh->q_odsc.bb.lb.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memset(subh->q_odsc.bb.ub.c, 0, sizeof(uint64_t) * BBOX_MAX_NDIM);
    memcpy(subh->q_odsc.bb.lb.c, lb, sizeof(uint64_t) * ndim);
    memcpy(subh->q_odsc.bb.ub.c, ub, sizeof(uint64_t) * ndim);
    strncpy(subh->q_odsc.name, var_name, strlen(var_name) + 1);

    // A hack to send our address to the server without using more space. This
    // field is ignored in a normal query.
    margo_addr_self(client->mid, &my_addr);
    margo_addr_to_string(client->mid, subh->q_odsc.owner, &owner_addr_size,
                         my_addr);
    margo_addr_free(client->mid, my_addr);

    in.odsc_gdim.size = sizeof(subh->q_odsc);
    in.odsc_gdim.raw_odsc = (char *)(&subh->q_odsc);
    in.param = subh->id;

    DEBUG_OUT("registered data subscription for %s with id %d\n",
              obj_desc_sprint(&subh->q_odsc), subh->id);

    set_global_dimension(&(client->dcg->gdim_list), var_name,
                         &(client->dcg->default_gdim), &od_gdim);
    in.odsc_gdim.gdim_size = sizeof(struct global_dimension);
    in.odsc_gdim.raw_gdim = (char *)(&od_gdim);

    get_server_address(client, &server_addr);

    hret = margo_create(client->mid, server_addr, client->sub_id, &handle);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_create() failed with %d.\n", __func__,
                hret);
        return (DSPACES_SUB_FAIL);
    }
    hret = margo_iforward(handle, &in, &req);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_forward() failed with %d.\n",
                __func__, hret);
        margo_destroy(handle);
        return (DSPACES_SUB_FAIL);
    }

    DEBUG_OUT("subscription %d sent.\n", subh->id);

    margo_addr_free(client->mid, server_addr);
    margo_destroy(handle);

    return (subh);
}

int dspaces_check_sub(dspaces_client_t client, dspaces_sub_t subh, int wait,
                      int *result)
{
    if(subh == DSPACES_SUB_FAIL) {
        fprintf(stderr,
                "WARNING: %s: status check on invalid subscription handle.\n",
                __func__);
        return DSPACES_SUB_INVALID;
    }

    DEBUG_OUT("checking status of subscription %d\n", subh->id);

    if(wait) {
        DEBUG_OUT("blocking on notification for subscription %d.\n", subh->id);
        ABT_mutex_lock(client->sub_mutex);
        while(subh->status == DSPACES_SUB_WAIT ||
              subh->status == DSPACES_SUB_RUNNING) {
            ABT_cond_wait(client->sub_cond, client->sub_mutex);
        }
        ABT_mutex_unlock(client->sub_mutex);
    }

    if(subh->status == DSPACES_SUB_DONE) {
        *result = subh->result;
    }

    return (subh->status);
}

static void kill_client_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_client_t client =
        (dspaces_client_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("Received kill message.\n");

    ABT_mutex_lock(client->drain_mutex);
    client->local_put_count = 0;
    ABT_cond_signal(client->drain_cond);
    ABT_mutex_unlock(client->drain_mutex);

    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(kill_client_rpc)

int dspaces_cancel_sub(dspaces_client_t client, dspaces_sub_t subh)
{
    if(subh == DSPACES_SUB_FAIL) {
        return (DSPACES_SUB_INVALID);
    }
    ABT_mutex_lock(client->sub_mutex);
    if(subh->status == DSPACES_SUB_WAIT) {
        subh->status = DSPACES_SUB_CANCELLED;
    }
    ABT_mutex_unlock(client->sub_mutex);

    return (0);
}

void dspaces_kill(dspaces_client_t client)
{
    uint32_t in;
    hg_addr_t server_addr;
    hg_handle_t h;
    margo_request req;
    hg_return_t hret;

    in = -1;

    DEBUG_OUT("sending kill signal to servers.\n");

    margo_addr_lookup(client->mid, client->server_address[0], &server_addr);
    hret = margo_create(client->mid, server_addr, client->kill_id, &h);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_create() failed\n", __func__);
        margo_addr_free(client->mid, server_addr);
        return;
    }
    margo_iforward(h, &in, &req);

    DEBUG_OUT("kill signal sent.\n");

    margo_addr_free(client->mid, server_addr);
    margo_destroy(h);
}
