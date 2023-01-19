#include <Visus/IdxDataset.h>
#include <vector>

struct idx1_dataset {
    idx1_dataset(Visus::SharedPtr<Visus::Dataset> dataset_) : dataset(dataset_) {}
    Visus::SharedPtr<Visus::Dataset> dataset;
};

extern "C" void idx1_init(int argc, char** argv)
{
    const char* idx_argv[] = {argv[0]};
    Visus::SetCommandLine(argc, idx_argv);
    Visus::DbModule::attach();
}

extern "C" void idx1_finalize()
{
    Visus::DbModule::detach();
}
extern "C" struct idx1_dataset* idx1_load_dataset(char* filepath)
{
    std::string fp(filepath);
    Visus::SharedPtr<Visus::Dataset> dataset = Visus::LoadDataset(fp);
    return (new struct idx1_dataset(dataset));
}

extern "C" size_t idx1_get_dtype_size(struct idx1_dataset *idset)
{
    return static_cast<size_t> (idset->dataset->getField().dtype.getByteSize());
}

extern "C" int idx1_get_max_resolution(struct idx1_dataset *idset)
{
    return idset->dataset->getMaxResolution();
}

extern "C" int idx1_get_ndims(struct idx1_dataset *idset)
{
    return idset->dataset->getLogicBox().p1.getPointDim();
}

extern "C" void idx1_get_lower_bound(struct idx1_dataset *idset, uint64_t *lb)
{
    int ndims = idset->dataset->getLogicBox().p1.getPointDim();
    if(!lb) {
        lb = (uint64_t*) malloc(ndims* sizeof(uint64_t));
    }
    for(int i=0; i<ndims; i++) {
        lb[i] = idset->dataset->getLogicBox().p1.get(i);
    }
}

extern "C" void idx1_get_upper_bound(struct idx1_dataset *idset, uint64_t *ub)
{
    int ndims = idset->dataset->getLogicBox().p2.getPointDim();
    if(!ub) {
        ub = (uint64_t*) malloc(ndims* sizeof(uint64_t));
    }
    for(int i=0; i<ndims; i++) {
        ub[i] = idset->dataset->getLogicBox().p2.get(i);
    }
}

extern "C" int idx1_get_field_num(struct idx1_dataset *idset)
{
    return static_cast<int> (idset->dataset->getFields().size());
}

extern "C" const char* idx1_get_field_name(struct idx1_dataset *idset, int fidx)
{
    return idset->dataset->getFields()[fidx].name.c_str();
}

extern "C" void idx1_get_timesteps(struct idx1_dataset *idset,
                                    int* ts_start, int* ts_end, int* ts_step)
{
    *ts_start = static_cast<int> (idset->dataset->getTimesteps().getRange().from);
    *ts_end = static_cast<int> (idset->dataset->getTimesteps().getRange().to);
    // ts_step is always 1?
    *ts_step = static_cast<int> (idset->dataset-> getTimesteps().getRange().step);
}

extern "C" void* idx1_read(idx1_dataset* idset, const char* fieldname, int ndims, 
                            size_t elemsize, uint64_t* lb, uint64_t* ub,
                            unsigned int ts, int resolution)
{
    std::string fn(fieldname);

    std::vector<Visus::Int64> v1, v2;
    v1.resize(ndims);
    v2.resize(ndims);
    for(int i=0; i<ndims; i++) {
        v1[i] = static_cast<Visus::Int64> (lb[i]);
        v2[i] = static_cast<Visus::Int64> (ub[i]);
    }
    Visus::PointNi p1(v1), p2(v2);

    Visus::BoxNi logic_box(p1, p2);

    int end_res = resolution == -1 ? idset->dataset->getMaxResolution() : resolution;
    double timestep = static_cast<double> (ts);

    //any time you need to read/write data from/to a Dataset create an Access
    auto access = idset->dataset->createAccess();
    auto field = !fn.empty() ? idset->dataset->getField(fn) : idset->dataset->getField();
    auto query = idset->dataset->createBoxQuery(logic_box, field, timestep, 'r');
    query->setResolutionRange(0, end_res);

    idset->dataset->beginBoxQuery(query);
    if(!query->isRunning()) {
        return NULL;
    }

    size_t data_size = elemsize;
    for(int i=0; i<ndims; i++) {
        data_size *= ub[i]-lb[i]+1;
    }
    //read data from disk
    if(!idset->dataset->executeBoxQuery(access, query)) {
        return NULL;
    }

    if(!query->buffer.c_size() == data_size) {
        return NULL;
    }

    return static_cast<void*>(query->buffer.c_ptr());
}