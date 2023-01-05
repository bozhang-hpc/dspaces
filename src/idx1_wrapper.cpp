#include <Visus/IdxDataset.h>
#include <vector>

extern "C" void* read_idx1(char* filepath, char* fieldname, int ndims,
                            size_t elemsize, uint64_t* lb, uint64_t* ub, 
                            unsigned int ts, int resolution)
{
    std::string fp(filepath), fn(fieldname);
    auto dataset = Visus::LoadDataset(fp);

    std::vector<Visus::Int64> v1, v2;
    v1.resize(ndims);
    v2.resize(ndims);
    for(int i=0; i<ndims; i++) {
        v1[i] = static_cast<Visus::Int64> (lb[i]);
        v2[i] = static_cast<Visus::Int64> (ub[i]);
    }
    Visus::PointNi p1(v1), p2(v2);

    Visus::BoxNi logic_box(p1, p2);

    int end_res = resolution == -1 ? dataset->getMaxResolution() : resolution;
    double timestep = static_cast<double> (ts); 

    //any time you need to read/write data from/to a Dataset create an Access
    auto access = dataset->createAccess();
    auto field = !fn.empty() ? dataset->getField(fn) : dataset->getField();
    auto query = dataset->createBoxQuery(logic_box, field, timestep, 'r');
    query->setResolutionRange(0, resolution);

    dataset->beginBoxQuery(query);
    if(!query->isRunning()) {
        return NULL;
    }

    size_t data_size = elemsize;
    for(int i=0; i<ndims; i++) {
        data_size *= ub[i]-lb[i]+1;
    }
    //read data from disk
    if(!dataset->executeBoxQuery(access, query)) {
        return NULL;
    }
    if(!query->buffer.c_size() == data_size) {
        return NULL;
    }

    return static_cast<void*>(query->buffer.c_ptr());
}