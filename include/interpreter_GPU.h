

/*Contract = GETT (out_modes Exp1 Exp2)
        | NestedGEMM (out_modes Exp1 Exp2)
        */

namespace CTL {
    
    std::shared_ptr<Output> CTL2GPU(std::shared_ptr<Output> Output);

    void expendOnGPU(std::shared_ptr<Output> program);
    void fuseOnGPU(std::shared_ptr<Output> program);
    void profiledFuseOnGPU(std::shared_ptr<Output> program);
    void prepareOnGPU(std::shared_ptr<Output> program); //must be invoked before compile
    void compileOnGPU(std::shared_ptr<Output> program);// must also allocate memory for the program
    CTL_stats statisticCTLOnGPU(std::shared_ptr<Output> program);
    void small_benchGPU();
    void evalOnGPU(std::shared_ptr<Output> program);
    void profiledEvalOnCPU(std::shared_ptr<Output> program);
    void ReleaseOnGPU(std::shared_ptr<Output> program);
    std::shared_ptr<Tensor::Tensor> getresultOnGPU(std::shared_ptr<Output> program);
    void allowtf32();
    void disallowtf32();
}
