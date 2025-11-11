#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include "ETL.h"

namespace ETL{

    struct MemAllocInfo
    {
        size_t offset; // byte offset in the big buffer
        size_t size;   // allocation size in bytes
        size_t birth;  // evaluation index when created
        size_t death;  // evaluation index when last used
    };

    // Greedy static memory planner
    class StaticMemoryPlanner
    {

    private:
        size_t alignment_;
        size_t eval_index_;
        std::vector<std::shared_ptr<Exp>> order_; // evaluation order
        std::vector<std::shared_ptr<Exp>> inputs; // inputs should be treated specially, because even it is in memry pool(GPU), it shouldn't be overwritten
        std::unordered_map<std::shared_ptr<Exp>, MemAllocInfo> node_info_;
        size_t scalar_size = 0; // size of scalar type in bytes
        size_t pool_size = 0;
        bool include_inputs;

        void postOrder(std::shared_ptr<Exp> node);
        void computeLifetimes();
        size_t assignOffsets();

        void updateDeath(std::shared_ptr<Exp> node, size_t user_time)
        {

            //input may not in, using [] would insert input
            auto it = node_info_.find(node);
            if (it != node_info_.end()){
                auto &info = it->second;
                info.death = std::max(info.death, user_time);
            }
            else
                return; // not tracked node

            if (auto p = std::dynamic_pointer_cast<Perm>(node)){
                auto it = node_info_.find(p->in_exp);
                if(it != node_info_.end()){
                    auto & kid_info = it->second;
                    kid_info.death = std::max(kid_info.death, user_time);
                }
            }
            else if (auto c = std::dynamic_pointer_cast<Contract>(node))
            {
                auto it = node_info_.find(c->in_exp1);
                if(it != node_info_.end()){
                    auto & kid1_info = it->second;
                    kid1_info.death = std::max(kid1_info.death, user_time);
                }
                it = node_info_.find(c->in_exp2);
                if(it != node_info_.end()){
                    auto & kid2_info = it->second;
                    kid2_info.death = std::max(kid2_info.death, user_time);
                }
            }
        }

        size_t alignUp(size_t v, size_t align) const
        {
            return (v + align - 1) / align * align;
        }
    public:
        StaticMemoryPlanner(size_t alignment = 128, bool include_inputs = true)
            : alignment_(alignment), include_inputs(include_inputs) {}

        void setAlignment(size_t alignment)
        {
            alignment_ = alignment;
        }

        void setScalarSize(size_t size)
        {
            scalar_size = size;
        }
        // Main entry: returns total buffer size needed
        size_t plan(std::shared_ptr<Output> prog)
        {
            eval_index_ = 0;
            node_info_.clear();
            order_.clear();

            // 1. Determine evaluation order (post-order)
            postOrder(prog->exp); //should put result tensor in the big workplace

            // 2. Compute lifetimes
            computeLifetimes();

            // 3. Allocate offsets greedily
            return assignOffsets();
        }

        // Query per-node offset
        size_t getOffset(const std::shared_ptr<Exp> node) const
        {
            auto it = node_info_.find(node);
            if(it == node_info_.end()) {
                //error
                throw std::runtime_error("Node not found in memory planner");
            }
            return it->second.offset;
        }

        bool incInput() const
        {
            return include_inputs;
        }
    };
};
#endif

