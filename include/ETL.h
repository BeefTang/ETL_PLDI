#ifndef ETL_H
#define ETL_H
#include <vector>
#include <memory>
#include <unordered_map>
#include "tensor.h"

/* syntax
Exp := Input (Tensor)
    | Permutation (out_modes in_modes Exp)
    | Contract ;later, each backend expand it into GETT_GPU or sth other

Contract = GEMM  (out_modes Exp1 Exp2)
        | GEMM (out_modes Exp1 Exp2)

ExpVal := Tensor

Tensor := a-tensor (modes a_continuous_mem)  ;input
*/
// the whole program(all exp including input) is evaluated in workplace, share a buig_buffer on GPU
// TODO: output tensor has an identical one

// finish first, then thinking how to fit better class abstrction
//optimize would be: CTL(only GETT) -> fine_CTL (perm+GETT+NestedGEMM) {similar to nameless_LET} -> machine speciifc CTL(add runting neeeded args and mems)

namespace ETL
{

    class Exp;
    class StaticMemoryPlanner;
    using ModeType = int;
    using Modes = std::vector<ModeType>;
    using SubscriptsType = std::vector<std::string>;


    enum DataPrecision :int {
        FP32 = 0,
        FP64 = 1,
        INT8 = 2,
        INT32 = 3,
        INT64 = 4
    };
    inline size_t Get_scalar_size(DataPrecision precision) {
        switch (precision)
        {
        case FP32:
            return sizeof(float);
        case FP64:
            return sizeof(double);
        case INT8:
            return sizeof(int8_t);
        case INT32:
            return sizeof(int32_t);
        case INT64:
            return sizeof(int64_t);
        default:
            throw std::invalid_argument("Unsupported data precision");
        }
    }

    // TODO: make int to ModeType
    class Context
    {
    public:
        std::unordered_map<std::string, ModeType> subscript_to_mode;
        std::unordered_map<ModeType, std::string> mode_to_subscript;
        std::unordered_map<ModeType, int64_t> mode_to_extent;

        ModeType next_mode = 0;              // cutensor needs int mode

        Context() = default;
        Context(const Context &other) = default;

        DataPrecision precision = FP32; // default precision
        void set_precision(DataPrecision p)
        {
            precision = p;
        }

        void set_mode_from_subscript(const std::string &subscript);
        void set_extent_from_mode(ModeType mode, int64_t extent);
        void set_extent_from_subscript(const std::string &subscript, int64_t extent);

        int64_t mode2extent(ModeType mode) const;
        std::vector<int64_t> mode2extent_v(const Modes &modes) const;

        std::string mode2subscript(const ModeType mode) const;
        SubscriptsType mode2subscript_v(const Modes modes) const;

        ModeType subscript2mode(const std::string &subscript) const;
        Modes subscript2mode_v(const std::vector<std::string> &subscripts) const;

        int64_t subscript2extent(const std::string &subscript) const;
        std::vector<int64_t> subscript2extent_v(const std::vector<std::string> &subscripts) const;
    };

    Modes extract_and_register_modes(
        const std::string &einsum_expr,
        const std::vector<int64_t> &dim_sizes,
        Context &ctx);

    std::pair<std::vector<Modes>, Modes> processing_einexp(const std::string &expr, const Context &ctx);


    struct CTL_stats {
        float mean_time_ms{0};
        float stddev_time_ms{0};
        float compile_time_ms{0};
        int num_gemms{0};
        int num_getts{0};
        int num_perms{0};
        //size_t max_ari_int{0};
        //size_t total_ari_int{0};
        //size_t max_intermediate_size{0};
        //size_t total_intermediate_size{0};
        //size_t total_flops{0};
        //size_t total_mem_access_bytes{0};

        void print() const{
            std::cout << "CTL statistics: " << std::endl;
            std::cout << "  Mean Time (ms): " << mean_time_ms << std::endl;
            std::cout << "  Stddev Time (ms): " << stddev_time_ms << std::endl;
            std::cout << "  Compile Time (ms): " <<compile_time_ms << std::endl;
            std::cout << "  Number of Permutations: " << num_perms << std::endl;
            std::cout << "  Number of GEMMs: " << num_gemms << std::endl;
            std::cout << "  Number of GETTs: " << num_getts << std::endl;
        }
    };

    class Exp
    {
    private:
        Modes out_modes;

    public:
        const Context &ctx;
        void *work_ptr{nullptr}; // pointer to the workspace, shared by all Exp in the program

        std::string name; // for print
        virtual void print(const std::string &prefix, bool is_left) const = 0;

    public:
        Exp(Modes out, const Context &ctx, std::string name) : out_modes(out), ctx(ctx), name(name) {}

        Exp(const Exp &other) : out_modes(other.out_modes), ctx(other.ctx) {}
        virtual ~Exp() = default;

        virtual void compile(){}; // interpreter to implement
        virtual void eval(){}; // interpreter to implement
        virtual void profiled_eval(int repeat=1){}; // interpreter to implement
        virtual CTL_stats ctl_statistics(){return CTL_stats{0,0,0};} // interpreter to implement
        float time_ms{0}; // time used in eval, for profiling
        float estimated_time_ms{0}; // time used in eval, for profiling
        bool profiled{false};


        std::vector<int> get_outmodes() const {
            return out_modes;
        }

        std::vector<int> get_modes() const
        {
            return out_modes;
        }

        std::vector<int64_t> get_extents() const
        {
            return ctx.mode2extent_v(out_modes);
        }

        SubscriptsType get_subscripts() const
        {
            return ctx.mode2subscript_v(out_modes);
        }

        void set_modes(Modes new_modes)
        {
            out_modes = new_modes;
        }

        size_t num_elements() const
        {
            size_t elements = 1;
            for (auto extent : get_extents())
            {
                elements *= extent;
            }
            return elements;
        }

    };

    class Perm : public Exp
    {
    public:
        std::shared_ptr<Exp> in_exp;
        Perm(Modes out_modes, const Context &ctx, std::shared_ptr<Exp> in, std::string name = "Perm") : Exp(out_modes, ctx, name), in_exp(in) { };
        
        void print(const std::string &prefix, bool is_left) const override;

        CTL_stats ctl_statistics() override{
            CTL_stats stats = in_exp->ctl_statistics();
            stats.num_perms += 1;
            return stats;
        }

    };

    class Contract : public Exp
    {
        public:
            std::shared_ptr<Exp> in_exp1, in_exp2;
            Contract(Modes out, const Context &ctx, std::shared_ptr<Exp> l, std::shared_ptr<Exp> r, std::string name = "Contract") : Exp(out, ctx, name), in_exp1(l), in_exp2(r) {};
            void print(const std::string &prefix, bool is_left) const override;

            double ari_int{0}; // used by its descendant
    };

    class Gemm : public Contract
    {
        public:
            Gemm(Modes out, const Context &ctx, std::shared_ptr<Exp> l, std::shared_ptr<Exp> r, std::string name = "GEMM") : Contract(out, ctx, l, r, name) {};

        public:
            size_t Im;
            size_t In;
            size_t Ik;
            size_t Ic;
    };

    class FusedTTGT : public Contract
    {
        public:
            FusedTTGT(Modes out, const Context &ctx, std::shared_ptr<Exp> l, std::shared_ptr<Exp> r, std::string name = "FusedTTGT") : Contract(out, ctx, l, r, name) {};
    };

    class Input : public Exp
    {
        // equal to ConstExp(tensor)
        public:
            const std::shared_ptr<Tensor::Tensor> tensor;
            Input(Modes out, const Context &ctx, std::shared_ptr<Tensor::Tensor> t, std::string name = "Input") : Exp(out, ctx, name), tensor(t) {}
            void print(const std::string &prefix, bool is_left) const override;

            CTL_stats ctl_statistics() override{
                CTL_stats stats;
                return stats;
            }
    };



    class Output // Program
    {
    public:
        Output() = default;
        Output(const Context &ctx) : ctx(ctx) {}

        Context ctx;                        // contex of the program
        std::shared_ptr<StaticMemoryPlanner> memory_planner = nullptr; // for static memory planning, pure program doesn't need
        /* program controls mempool, because it may point to CPU or GPU. mem planner should be independent of  */
        void *big_buffer{nullptr}; // for static memory planning
        //void *output_ptr{nullptr}; // pointer to the output tensor, won't be released by program
        std::shared_ptr<Exp> exp{nullptr};

        void print() const;
        void expend();

    private:
        bool expanded = false;


    };


    /************
    ETL tree Utilities

    *******/
    std::shared_ptr<Output> build_ETL_tree(std::string &expr, std::vector<int64_t> &sizes, std::vector<std::pair<int, int>> &contraciton_path, std::vector<void *> *tensors=nullptr, DataPrecision precision = FP32);

    std::shared_ptr<Exp> Expand(std::shared_ptr<Exp> exp);
}

#endif
