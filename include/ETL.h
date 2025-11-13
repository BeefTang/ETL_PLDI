#ifndef ETL_H
#define ETL_H
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include "tensor.h"
#include "context.h"

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
//optimize would be: ETL(only GETT) -> fine_ETL (perm+GETT+NestedGEMM) {similar to nameless_LET} -> machine speciifc ETL(add runting neeeded args and mems)

namespace ETL
{

    class Exp;
    class StaticMemoryPlanner;


    struct ETL_stats {
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
            std::cout << "ETL statistics: " << std::endl;
            std::cout << "  Mean Time (ms): " << mean_time_ms << std::endl;
            std::cout << "  Stddev Time (ms): " << stddev_time_ms << std::endl;
            std::cout << "  Compile Time (ms): " <<compile_time_ms << std::endl;
            std::cout << "  Number of Permutations: " << num_perms << std::endl;
            std::cout << "  Number of GEMMs: " << num_gemms << std::endl;
            std::cout << "  Number of GETTs: " << num_getts << std::endl;
        }
    };

    /*****************
     Ued by DP
     ****************************/
     //E_node have to check its constrain's up_L, K,C is compitable with possible(down_M N C) of the exp of this node  
    struct E_node{
        enum {input, nochild, onechild, twochilren} kind; //threr only one nochild which pick the optimal children from each children list
        //Modes up_L, up_K, up_C;//constrains for father constrain, if 
        union{
            std::shared_ptr<Constrain> input; //only up_ filled
            struct {
                std::shared_ptr<Constrain> constrain;
            } one;
            struct {
                std::shared_ptr<Constrain> constrain1, constrain2; 
            } two;

        } u;

        // all are ordered
        int num_perms=0;
    };

    struct nodeCmp{
        bool operator()(std::shared_ptr<E_node> a, std::shared_ptr<E_node> b) const {
            return a->num_perms < b->num_perms;
        }
    };

    struct Constrain{
        Modes whole;
        Modes up_L_ordered, up_K_ordered, up_C_ordered;//K is not used for up and down, but for brother
        Modes down_M_ordered, down_N_ordered, down_C_ordered;
    };

    /*********************************************/

    class Exp
    {
    private:
        Modes out_modes;

    public:
        std::unordered_set<ModeType> down_M, down_N, down_K, down_C; //not ordered
        std::unordered_set<ModeType> up_L, up_C, up_K;


        std::set<std::shared_ptr<E_node>, nodeCmp> opt_list;
        std::vector<std::shared_ptr<Constrain>> possible_list;

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
        virtual ETL_stats ctl_statistics(){return ETL_stats{0,0,0};} // interpreter to implement
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
            Modes in_modes;
        
        public:
            std::shared_ptr<Exp> in_exp;
            Perm(Modes out_modes, const Context &ctx, std::shared_ptr<Exp> in, std::string name = "Perm") : Exp(out_modes, ctx, name), in_exp(in) { };
            
            void print(const std::string &prefix, bool is_left) const override;

            ETL_stats ctl_statistics() override{
                ETL_stats stats = in_exp->ctl_statistics();
                stats.num_perms += 1;
                return stats;
            }

    };

    class Contract : public Exp
    {
        //public:
        //    Modes l_in_modes, r_in_modes;

        public:
            std::shared_ptr<Exp> in_exp1, in_exp2;
            Contract(Modes out, const Context &ctx, std::shared_ptr<Exp> l, std::shared_ptr<Exp> r, std::string name = "Contract") : Exp(out, ctx, name), in_exp1(l), in_exp2(r) {};
            void print(const std::string &prefix, bool is_left) const override;

            double ari_int{0}; // used by its descendant
    };

    class Gemm : public Contract
    {
        public:
            //Modes are also E-class (child of the node), used to identify E-node
            //Modes M, N, K, C; // out_modes==MNC 
            bool l_trans, r_trans; //l_in_modes == MK if l_trans=false, else l_in_modes==KM 

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

            ETL_stats ctl_statistics() override{
                ETL_stats stats;
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
    void Esat(std::shared_ptr<Output> program);
}

#endif
