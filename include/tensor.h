#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
namespace Tensor
{
    class Tensor
    {
        /* Tensor is Modes + a continnuous mem
        */
        // as value of CTL without machine-specific details, so only CPU_mem is needed
    public:
        Tensor(void *data, std::vector<int64_t> shape_, bool synthetic=false)
            : ptr(data), shape(shape_), synthetic(synthetic) {}

        // copy
        Tensor(const Tensor &other)
            :  ptr(other.ptr), shape(other.shape)   // copying data pointer, do not need to copy synthetic, only the original tensor in charge to release it
        { 
            if (other.synthetic){
                std::cerr << "Warning: copying a synthetic tensor, copied tensor may be released"<<ptr<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            else{
                ptr = other.ptr;
                shape = other.shape;
            }
        }


        ~Tensor(){
            if(synthetic && ptr!=nullptr){
                //std::cout<<"Releasing synthetic tensor data: "<<ptr<<std::endl;
                delete[] static_cast<char*>(ptr); // assuming float for synthetic
                ptr = nullptr;
            }
        }

        std::vector<int64_t> get_shape() const{
            return shape;
        }
        void *get_ptr() const{
            return ptr;
        }


    private:
        void* ptr{nullptr};
        std::vector<int64_t> shape;
        bool synthetic{false};

    };
}

#endif