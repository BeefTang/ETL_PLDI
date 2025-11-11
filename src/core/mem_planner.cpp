#include "mem_planner.h"


namespace ETL{

    // --------- Step 1: evaluation order ----------
    void StaticMemoryPlanner::postOrder(std::shared_ptr<Exp> node) {

        if (auto p = std::dynamic_pointer_cast<Perm>(node)){
            
            postOrder(p->in_exp);
            order_.push_back(node);
        }
        else if (auto c = std::dynamic_pointer_cast<Contract>(node)) {
            postOrder(c->in_exp1);
            postOrder(c->in_exp2);
            order_.push_back(node);
        }
        // Input has no children
        else if (auto i = std::dynamic_pointer_cast<Input>(node)) {
            if(include_inputs) { //needs extra memory for inputs for example: GPU device memory 
                inputs.push_back(node);
            }
        }
    }


    // --------- Step 2: lifetimes ----------
    void StaticMemoryPlanner::computeLifetimes() {
        // Birth time is just evaluation order index
        for (size_t i = 0; i < order_.size(); ++i) {
            auto e = order_[i];
            MemAllocInfo info;
            info.birth = i;
            info.death = i; // will extend later
            info.size = e->num_elements() * scalar_size; // <-- adjust type
            //info.size = alignUp(info.size, alignment_);
            node_info_[e] = info;
        }

        // Death = last use in parent evaluation
        for (size_t i = 0; i < order_.size(); ++i) {
            auto e = order_[i];
            updateDeath(e, i);
        }

    }


    // --------- Step 3: greedy offset assignment ----------
    size_t StaticMemoryPlanner::assignOffsets()
    {
        size_t total_size = 0;
        for(auto &input : inputs){
            //inputs are never overwritten, so their death is max
            MemAllocInfo info;
            info.birth = 0;
            info.death = order_.size(); // last forever
            info.size = input->num_elements() * scalar_size;
            info.offset = total_size;
            total_size = alignUp(total_size + info.size, alignment_);
            node_info_[ input ] = info;
        }

        // then processing meddile nodes
        struct FreeBlock
        {
            size_t offset, size;
        }; // offset is start of the block, imitating runtime memory allocation

        std::vector<FreeBlock> free_list;

        // sort by birth time
        std::vector<std::shared_ptr<Exp>> sorted = order_;

        std::sort(sorted.begin(), sorted.end(),
                  [&](std::shared_ptr<Exp> a, std::shared_ptr<Exp> b)
                  {
                      return node_info_[a].birth < node_info_[b].birth;
                  });


        struct Active
        {
            size_t death;
            size_t offset;
            size_t size;
        };
        std::vector<Active> active_blocks;

        for (auto &exp : sorted)
        {
            auto &info = node_info_[exp];
            int t = info.birth;

            // 1. Free expired nodes
            for (auto it = active_blocks.begin(); it != active_blocks.end();)
            {
                if (it->death < t)
                {
                    free_list.push_back({it->offset, it->size});
                    it = active_blocks.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // 2. Find best-fit free block
            auto fit = free_list.end();
            for (auto it = free_list.begin(); it != free_list.end(); ++it)
            {
                if (it->size >= info.size)
                {
                    if (fit == free_list.end() || it->size < fit->size) // find smallest fit
                        fit = it;
                }
            }

            size_t offset_of_active = 0, size_of_active = 0;

            if (fit != free_list.end())
            {
                // Reuse block
                offset_of_active = fit->offset;

                // Shrink block
                auto end = fit->offset + fit->size;
                size_t aligned = alignUp(offset_of_active + info.size, alignment_); // keep offset always aligned
                if (end <= aligned)
                {
                    // the remaining block can not exist with offset satisfying alignment. remove, and assign the whole block to this exp
                    size_of_active = fit->size;
                    free_list.erase(fit);
                }
                else
                {
                    fit->offset = aligned;
                    fit->size = end - aligned;
                    size_of_active = aligned - offset_of_active;
                }
            }
            else
            {
                // Allocate new space at end
                offset_of_active = alignUp(total_size, alignment_);
                size_of_active = info.size;
                total_size = offset_of_active + size_of_active;
            }

            info.offset = offset_of_active;
            active_blocks.push_back({info.death, offset_of_active, size_of_active});
        }

        //for(auto [exp, info] : node_info_){
        //    std::cout<<"Node "<<exp->name<<", size "<<info.size<<", offset "<<info.offset<<", birth "<<info.birth<<", death "<<info.death<<std::endl;
        //}

        //std::cout << "Total memory allocated for this contraction = " << total_size << "\n";

        return total_size;
    }
};
