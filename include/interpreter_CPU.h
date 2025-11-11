
/*Contract = GETT (out_modes Exp1 Exp2)
        | NestedGEMM (out_modes Exp1 Exp2)
        */

namespace CTL {
    
    std::shared_ptr<Output> CTL2CPU(std::shared_ptr<Output> Output);

    void expendOnCPU(std::shared_ptr<Output> program);
    void fuseOnCPU(std::shared_ptr<Output> program);
    void profiledFuseOnCPU(std::shared_ptr<Output> program);
    void prepareOnCPU(std::shared_ptr<Output> program); //must be invoked before compile
    void compileOnCPU(std::shared_ptr<Output> program);// must also allocate memory for the program
    CTL_stats statisticCTLOnCPU(std::shared_ptr<Output> program);
    void small_benchCPU();
    void evalOnCPU(std::shared_ptr<Output> program);
    void profiledEvalOnCPU(std::shared_ptr<Output> program);
    void ReleaseOnCPU(std::shared_ptr<Output> program);
    std::shared_ptr<Tensor::Tensor> getresultOnCPU(std::shared_ptr<Output> program);
}
