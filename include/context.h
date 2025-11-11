#ifndef CTX_H
#define CTX_H

namespace ETL{
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
};

#endif
