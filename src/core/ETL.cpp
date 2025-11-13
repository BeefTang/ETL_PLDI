#include <iostream>
#include <cassert>
#include <unordered_set>
#include <regex>
#include <random>

#include "ETL.h"
#include "mem_planner.h"
#include "dimensions_utils.h"
#include "solve_perms.h"

namespace ETL {


    /********************************************
    *
    Contraction Tree Utilities
    *
    *******************************************/
    Modes contract_modes(const Modes &v1, const Modes &v2, const Modes &final_out)
    {
        //Important if some mode is in final_out, which means it is C dimenison, and shouldedn't be reduced
        std::unordered_set<ModeType> set1(v1.begin(), v1.end());
        std::unordered_set<ModeType> set2(v2.begin(), v2.end());
        std::unordered_set<ModeType> set_final(final_out.begin(), final_out.end());
        Modes result;

        for (const auto &s : v1)
        {
            if (set2.find(s) == set2.end()){
                //M or N dim, should be in result
                result.push_back(s);
            }
            else if(set_final.find(s) != set_final.end()) {
                // C dim
                result.push_back(s);
            }
        }
        for (const auto &s : v2)
        {
            if (set1.find(s) == set1.end()){
                result.push_back(s);
            }
            else if(set_final.find(s) != set_final.end()) {
                // C dim
                result.push_back(s);
            }
        }

        //for(auto &s: result) {
        //    std::cout<<s<<","; // mode should be non-negative
        //}
        //std::cout<<std::endl;

        shuffle(result.begin(), result.end(), std::mt19937{42}); // random shuffle the modes to avoid bias

        //for(auto &s: result) {
        //    std::cout<<s<<","; // mode should be non-negative
        //}
        //std::cout<<std::endl;
        return result;
    }



    std::shared_ptr<Exp> construct_tree_from_given_path(std::pair<std::vector<Modes>, Modes> processed_exp,
                                                           const std::vector<void *> *input_pointers,
                                                           const std::vector<std::pair<int, int>> &given_path,
                                                           Context &ctx)
    {

        std::vector<std::shared_ptr<Exp>> treelist;
        if (input_pointers == nullptr) {
            // make dummy tensor, if not provided
            std::cout<<"Generated tensors data used for benchmark:"<<std::endl;
            for (const auto &modes : processed_exp.first) {
                auto extents = ctx.mode2extent_v(modes);

                int64_t num_elements = std::accumulate(
                    extents.begin(), extents.end(), int64_t{1},
                    [](int64_t a, int64_t b)
                    { return a * b; });

                void *data = static_cast<void *>(new char[num_elements * Get_scalar_size(ctx.precision)]); // allocate raw memory


                if (ctx.precision == FP32) {
                    float *fdata = static_cast<float *>(data);
                    for (int64_t i = 0; i < num_elements; i++) {
                        fdata[i] = static_cast<float>(i % 16); // some varying data
                    }
                } else if (ctx.precision == FP64) {
                    double *ddata = static_cast<double *>(data);
                    for (int64_t i = 0; i < num_elements; i++) {
                        ddata[i] = static_cast<double>(i % 16); // some varying data
                    }
                } else if (ctx.precision == INT8) {
                    int8_t *idata = static_cast<int8_t *>(data);
                    for (int64_t i = 0; i < num_elements; i++) {
                        idata[i] = static_cast<int64_t>(i % 16); // some varying data
                    }
                } else if (ctx.precision == INT32) {
                    int32_t *idata = static_cast<int32_t *>(data);
                    for (int64_t i = 0; i < num_elements; i++) {
                        idata[i] = static_cast<int32_t>(i % 16); // some varying data
                    }
                } else if (ctx.precision == INT64) {
                    int64_t *idata = static_cast<int64_t *>(data);
                    for (int64_t i = 0; i < num_elements; i++) {
                        idata[i] = static_cast<int64_t>(i % 16); // some varying data
                    }
                } else {
                    throw std::invalid_argument("Unsupported data precision");
                }

                /********* */
                //std::cout<<data<<": "<<num_elements * Get_scalar_size(ctx.precision)<<" bytes"<<std::endl;
                /********* */
                
                treelist.push_back(
                    std::make_shared<Input>(
                        modes, ctx,
                        std::make_shared<Tensor::Tensor>(data, ctx.mode2extent_v(modes), true)));
            }
        }
        else
        {
            for (int i = 0; i < processed_exp.first.size(); i++)
            {
                auto modes = processed_exp.first[i];
                treelist.push_back(
                    std::make_shared<Input>(
                        modes, ctx,
                        std::make_shared<Tensor::Tensor>((*input_pointers)[i], ctx.mode2extent_v(modes))));
            }
        }

        auto final_out_modes = processed_exp.second;
        for (auto contract : given_path)
        {
            auto &l = treelist[contract.first];
            auto &r = treelist[contract.second];
            auto new_modes = contract_modes(l->get_outmodes(), r->get_outmodes(), final_out_modes);

            auto temp = std::make_shared<Contract>(new_modes, ctx, treelist[contract.first], treelist[contract.second]);

            /**********************/
            auto l_dims = l->get_modes();
        
            auto r_dims = r->get_modes();
            auto this_dims = new_modes;


            auto mnkc = get_M_N_K_C(l_dims, r_dims, this_dims);


            temp->down_M.insert(mnkc[0].begin(), mnkc[0].end());
            temp->down_N.insert(mnkc[1].begin(), mnkc[1].end());
            temp->down_K.insert(mnkc[2].begin(), mnkc[2].end());
            temp->down_C.insert(mnkc[3].begin(), mnkc[3].end());

            l->up_L.insert(mnkc[0].begin(), mnkc[0].end());
            l->up_K.insert(mnkc[2].begin(), mnkc[2].end());
            l->up_C.insert(mnkc[3].begin(), mnkc[3].end());

            r->up_L.insert(mnkc[1].begin(), mnkc[1].end());
            r->up_K.insert(mnkc[2].begin(), mnkc[2].end());
            r->up_C.insert(mnkc[3].begin(), mnkc[3].end());
            /*****************/

            int big = contract.first > contract.second ? contract.first : contract.second;
            int small = contract.first > contract.second ? contract.second : contract.first;
            treelist.erase(treelist.begin() + big); // erase big first, no effect to small one's position
            treelist.erase(treelist.begin() + small);
            treelist.push_back(temp);
        }

        treelist[0]->set_modes(final_out_modes); // ensure same output modes order
        return treelist[0];
    }



//modes and CTX functionality 
std::vector<int> extract_and_register_modes(
    const std::string& einsum_expr,
    const std::vector<int64_t>& dim_sizes,
    Context& ctx
) {
    std::vector<std::string> subscripts;  // Unique subscripts
    std::unordered_set<std::string> subscript_set;

    bool is_number_based = einsum_expr.find('[') != std::string::npos;

    if (is_number_based) {
        // Match patterns like [26,27]
        std::regex bracket_expr(R"(\[(.*?)\])");
        std::smatch match;
        std::string::const_iterator search_start(einsum_expr.cbegin());

        while (std::regex_search(search_start, einsum_expr.cend(), match, bracket_expr)) {
            std::stringstream ss(match[1]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                std::string sub = std::to_string(std::stoi(token)); // Clean whitespace
                if (subscript_set.insert(sub).second) {
                    subscripts.push_back(sub);
                }
            }
            search_start = match.suffix().first;
        }

        // Sort by numerical order
        std::sort(subscripts.begin(), subscripts.end(),
                  [](const std::string& a, const std::string& b) {
                      return std::stoi(a) < std::stoi(b);
                  });

    } else {
        // Character-based, assume format like "abc,bfg->afg"
        for (char c : einsum_expr) {
            if (std::isalpha(c)) {
                std::string s(1, c);
                if (subscript_set.insert(s).second) {
                    subscripts.push_back(s);
                }
            }
        }

        // Sort alphabetically
        std::sort(subscripts.begin(), subscripts.end());
    }

    Modes modes;
    for (const std::string& sub : subscripts) {
        ctx.set_mode_from_subscript(sub);
        modes.push_back(ctx.subscript2mode(sub));
        if (is_number_based) {
            int idx = std::stoi(sub);
            if (idx >= 0 && idx < static_cast<int>(dim_sizes.size())) {
                ctx.set_extent_from_subscript(sub, dim_sizes[idx]);
            } else {
                std::cerr << "Index " << idx << " out of range in dim_sizes.\n";
                exit(1);
            }
        }
    }

    return modes;
}

std::pair<std::vector<Modes>, Modes> processing_einexp(const std::string& expr, const Context& ctx) {
    bool is_number_based = expr.find('[') != std::string::npos;

    std::vector<std::vector<int>> input_modes;
    std::vector<int> output_modes;

    std::string input_part, output_part;
    auto arrow_pos = expr.find("->");
    if (arrow_pos != std::string::npos) {
        input_part = expr.substr(0, arrow_pos);
        output_part = expr.substr(arrow_pos + 2);
    } else {
        input_part = expr;
        output_part = "";  // No output specified
    }

    if (is_number_based) {
        // Process input tensors
        std::regex bracket_expr(R"(\[(.*?)\])");
        std::smatch match;
        std::string::const_iterator search_start(input_part.cbegin());

        while (std::regex_search(search_start, input_part.cend(), match, bracket_expr)) {
            std::stringstream ss(match[1]);
            std::string token;
            std::vector<int> tensor_modes;
            while (std::getline(ss, token, ',')) {
            //    std::string sub = std::to_string(std::stoi(token));
                int mode = ctx.subscript2mode(token);
                tensor_modes.push_back(mode);
            }
            input_modes.push_back(tensor_modes);
            search_start = match.suffix().first;
        }

        // Process output
        if (!output_part.empty()) {
            std::regex output_expr(R"(\[(.*?)\])");
            if (std::regex_search(output_part, match, output_expr)) {
                std::stringstream ss(match[1]);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    //std::string sub = std::to_string(std::stoi(token));
                    output_modes.push_back(ctx.subscript2mode(token));
                }
            }
        }

    } else {
        // Alphabetic: comma-separated input, output is string
        std::stringstream ss(input_part);
        std::string term;
        while (std::getline(ss, term, ',')) {
            std::vector<int> tensor_modes;
            for (char c : term) {
                if (std::isalpha(c)) {
                    std::string sub(1, c);
                    tensor_modes.push_back(ctx.subscript2mode(sub));
                }
            }
            input_modes.push_back(tensor_modes);
        }

        for (char c : output_part) {
            if (std::isalpha(c)) {
                std::string sub(1, c);
                output_modes.push_back(ctx.subscript2mode(sub));
            }
        }
    }

    return {input_modes, output_modes};
}




    /*********************************************************
    ETL tree Utilities
    *********************************************************/
std::shared_ptr<Output> build_ETL_tree(std::string &expr, std::vector<int64_t> &sizes, std::vector<std::pair<int, int>> &contraciton_path, std::vector<void *> *tensors, DataPrecision precision)
    {
        auto program = std::make_shared<Output>();
        program->ctx.set_precision(precision); // set the precision for the context
        // prpare context's mode-subscript mapping first
        auto ordered_modes = extract_and_register_modes(expr, sizes, program->ctx);
        assert(ordered_modes.size() == sizes.size());
        for (int i = 0; i < ordered_modes.size(); i++)
        {
            program->ctx.set_extent_from_mode(ordered_modes[i], sizes[i]);
        }
        auto processed_exp = processing_einexp(expr, program->ctx);

        program->exp = construct_tree_from_given_path(processed_exp, tensors, contraciton_path, program->ctx);

        return program;
    }

    /********* Expand ********/
    size_t product(const std::vector<int64_t> &dims) {
        return std::accumulate(dims.begin(), dims.end(), 1UL, std::multiplies<size_t>());
    }

    double arithmetic_intensity(size_t Im, size_t In, size_t Ik, size_t Ic, size_t scalarSize ) {
        size_t flops = 2 * Im * In * Ik * Ic;

        // Bytes moved (assuming float32)
        size_t bytes = scalarSize* (Im * Ik * Ic + In * Ik * Ic + Im * In * Ic);

        return static_cast<double>(flops) / static_cast<double>(bytes);
    }

    std::shared_ptr<Exp> Expand_exp(std::shared_ptr<Exp> exp){

        //termination
        if (auto i = std::dynamic_pointer_cast<Input>(exp))
        {
            return exp;
        }
        else if (auto c = std::dynamic_pointer_cast<FusedTTGT>(exp))
        {
            // Bottom to top recursive
            auto new_l = Expand_exp(c->in_exp1);
            auto new_r = Expand_exp(c->in_exp2);

            // KNOWn:now l and r must be canonical, and both having input or contraciton as root
            auto l_dims = new_l->get_modes();
        
            auto r_dims = new_r->get_modes();
            auto this_dims = exp->get_modes();
            if (!is_already_ordered(l_dims, r_dims, this_dims))
            {
                reorder_all(l_dims, r_dims, this_dims);
                if (l_dims != new_l->get_modes())
                {
                    new_l = std::make_shared<Perm>(l_dims, c->ctx, new_l);
                }
                if (r_dims != new_r->get_modes())
                {
                    new_r = std::make_shared<Perm>(r_dims, c->ctx, new_r);
                }
            }

            std::shared_ptr<Gemm> ret = std::make_shared<Gemm>(this_dims, c->ctx, new_l, new_r);

            auto mnkc = get_M_N_K_C(l_dims, r_dims, this_dims);
            ret->Im = product(c->ctx.mode2extent_v(mnkc[0]));
            ret->In = product(c->ctx.mode2extent_v(mnkc[1]));
            ret->Ik = product(c->ctx.mode2extent_v(mnkc[2]));
            ret->Ic = product(c->ctx.mode2extent_v(mnkc[3]));
            ret->ari_int = arithmetic_intensity(ret->Im, ret->In, ret->Ik, ret->Ic, Get_scalar_size(c->ctx.precision));

            //ret->cache.M = ret->Im;
            //ret->cache.N = ret->In;
            //ret->cache.K = ret->Ik;
            //ret->cache.C = ret->Ic;

            return ret;

        }
        else
        {
            std::cerr << "Fatal: permutation node should no be optimized\n";
            std::exit(EXIT_FAILURE);
        }
    }


    /***********E-saturation*************** */ 

    bool check_order(
            const std::unordered_set<ModeType>& a,
            const std::unordered_set<ModeType>& b,
            const std::unordered_set<ModeType>& c,
            const std::vector<ModeType>& d,     // keep ordered
            const std::vector<char>& order)
    {
        // 1. Build index map of d
        std::unordered_map<ModeType, int> pos;
        for (int i = 0; i < (int)d.size(); ++i)
            pos[d[i]] = i;

        // 2. Compute [min, max] index range for each group
        auto get_range = [&](const std::unordered_set<ModeType>& v) {
            int min_idx = (int)d.size();
            int max_idx = -1;
            for (auto x : v) {
                auto it = pos.find(x);
                if (it == pos.end()) continue; // ignore elements not in d
                int p = it->second;
                min_idx = std::min(min_idx, p);
                max_idx = std::max(max_idx, p);
            }
            return std::pair<int, int>{min_idx, max_idx};
        };

        auto range_a = get_range(a);
        auto range_b = get_range(b);
        auto range_c = get_range(c);

        // Helper to access by char
        auto get_range_by_name = [&](char name) -> std::pair<int, int> {
            if (name == 'a') return range_a;
            if (name == 'b') return range_b;
            return range_c;
        };

        // 3. Check that order[i]’s range finishes before order[i+1] starts
        for (size_t i = 0; i + 1 < order.size(); ++i) {
            auto r1 = get_range_by_name(order[i]);
            auto r2 = get_range_by_name(order[i + 1]);
            if (r1.second >= r2.first) return false;
        }
        return true;
    }

    //given whole order, and a subset, give orderd of that subset
    Modes ordering_subset(const Modes & order, const std::unordered_set<ModeType> &s){
        Modes result;
        result.reserve(s.size());  // optional

        for (ModeType x : order) {
            if (s.count(x)) {  // check if element is in the set
                result.push_back(x);
            }
        }
        return result;
    }

    bool is_compitable_b(const std::shared_ptr<Constrain> down, const std::shared_ptr<Constrain> up){
            //TODO: check if input == from_possible.down_M_ordered/N + from_possible.down_C_ordered
            if((up->up_L_ordered == down->down_N_ordered || 
                        up->up_L_ordered == down->down_M_ordered) && 
                    up->up_C_ordered == down->down_C_ordered){
                return true;
            }
            else{
                return false;
            }
    }
    bool is_compitable(const std::shared_ptr<Constrain> from_possible, const std::shared_ptr<E_node> node){
        if(node->kind == input) {
            //TODO: check if input == from_possible.down_M_ordered/N + from_possible.down_C_ordered
            //if((node->u.input.up_L_ordered == from_possible->down_N_ordered || 
            //            node->u.input.up_L_ordered == from_possible->down_M_ordered) && 
            //        node->u.input.up_C_ordered == from_possible->down_C_ordered){
            if(is_compitable_b(from_possible, node->u.input)){
                return true;
            }
            else{
                return false;
            }
        }
        else if(node->kind == nochild)
            return true;
        else if(node->kind == onechild){
            if(is_compitable_b(from_possible, node->u.one.constrain)){
                return true;
            }
            else{
                return false;
            }
        }
        else if(node->kind == twochilren){
            if(is_compitable_b(from_possible, node->u.two.constrain1) &&
               is_compitable_b(from_possible, node->u.two.constrain2)){
                return true;
            }
            else{
                return false;
            }
        }

        
    }

    void DP_saturation(std::shared_ptr<Exp> exp, std::unordered_set<ModeType>& hotpoints){

        if (auto i = std::dynamic_pointer_cast<Input>(exp))
        {
                Modes leaf = i->get_outmodes();
                bool L_before_K =  check_order(i->up_L, i->up_K, i->up_C, leaf, {'a', 'b', 'c'});
                bool K_before_L =  check_order(i->up_L, i->up_K, i->up_C, leaf, {'b', 'a', 'c'});
                //update possible_list
                auto temp = std::make_shared<Constrain>();
                if(L_before_K || K_before_L){
                    temp->whole = leaf; 
                    temp->up_L_ordered = ordering_subset(leaf,  i->up_L); temp->up_K_ordered = ordering_subset(leaf,  i->up_K); temp->up_C_ordered =ordering_subset(leaf,  i->up_C);   
                    //  leaf doean't have down temp->down_M_ordered = ordering_subset(leaf,  i->down_M); temp->down_N_ordered = ordering_subset(leaf,  i->down_N); temp->down_C_ordered =ordering_subset(leaf,  i->down_C); 
                    i->possible_list.push_back(temp);
                }

                //update opt_list
                auto temp_node = std::make_shared<E_node>();
                temp_node->kind = input;
                temp_node->u.input = temp;//constrain is itself
                temp_node->num_perms = 0;
                i->opt_list.insert(temp_node);
            return;
        }
        else if (auto c = std::dynamic_pointer_cast<Contract>(exp))
        {
            // Bottom to top recursive
            DP_saturation(c->in_exp1, hotpoints);
            DP_saturation(c->in_exp2, hotpoints);

            Modes S = c->get_outmodes();
            if(c->up_L.empty()){//root contract
                                //check if set M and N
                if(check_order(c->down_M, c->down_N, c->down_C, S, {'a', 'b', 'c'}) || check_order(c->down_M, c->down_N, c->down_C, S, {'b', 'a', 'c'})){
                    std::cout<<"root is gemm already"<<std::endl;
                }
                else{
                    std::cout<<"not possible"<<std::endl;
                }

            }
            else{
                std::vector<std::pair<std::vector<UnorderedModes>, UnorderedModes>> rules = {
                        {{c->down_M, c->down_N}, c->down_C},
                        {{c->up_L, c->up_K}, c->up_C}
                    };

                auto perms = permutation_with_rules(S, rules);
                std::cout << "Number of valid permutations: " << perms.size() << "\n";
                for (auto &p : perms) {
                    auto temp = std::make_shared<Constrain>();
                    temp->whole = p; 
                    temp->down_M_ordered = ordering_subset(p,  c->down_M); temp->down_N_ordered = ordering_subset(p,  c->down_N); temp->down_C_ordered =ordering_subset(p,  c->down_C);   
                    temp->up_L_ordered = ordering_subset(p,  c->up_L); temp->up_K_ordered = ordering_subset(p,  c->up_K); temp->up_C_ordered =ordering_subset(p,  c->up_C);     

                    c->possible_list.push_back(temp);
                }
            }

            // update opt_list
            if(c->in_exp1->possible_list.empty() && c->in_exp2->possible_list.empty()){
                //find opt_list's best to form a no children
                auto temp = std::make_shared<E_node>();
                temp->kind = nochild;
                temp->num_perms = c->in_exp1->opt_list.begin()->num_perms + c->in_exp2->opt_list.begin()->num_perms + 2;
                c->opt_list.push_back(temp);
            }
            else if(!(c->in_exp1->possible_list.empty()) && !(c->in_exp2->possible_list.empty())){
                for(auto &x:c->in_exp1->possible_list){
                }
                for(auto & x:c->in_exp1->possible_list){
                    for(auto &y:c->in_exp2->possible_list){
                        //check if they have same up_K
                        if(x->up_K_ordered == y->up_K_ordered && x->up_C_ordered == y->up_C_ordered){
                            auto temp = std::make_shared<E_node>();
                            temp->kind = twochilren;
                            temp->u.two.constrain1 = x;
                            temp->u.two.constrain2 = y;
                            temp->num_perms = c->in_exp1->opt_list.begin() + c->in_exp2->opt_list.begin();
                            c->opt_list.push_back(temp);
                        }
                        else{
                            //one of the child must perm
                            auto temp1 = std::make_shared<E_node>();
                            temp1->kind = onechild;
                            temp1->u.one.constrain = x;
                            temp->num_perms = c->in_exp1->opt_list.begin() + c->in_exp2->opt_list.begin() +1;
                            c->opt_list.push_back(temp1);
                        }
                            E_node temp;
                            temp.kind = twochilren;
                            for(auto & t:c->exp1->opt_list){
                                //then find in opt_list for its cost
                                if(is_compitable(x, t.
                                
                            }
                            temp.num_perms = c->exp1->opt_list.begin() + c->exp2->opt_list.begin();
                            c->opt_list.push_back(temp);
                        }
                    }
                }
            }
            else if(!(c->in_exp1->possible_list.empty()) || !(c->in_exp2->possible_list.empty())){
                //if one is not empty, 
                for(auto x:c->in_exp1->possible_list){
                    E_node temp;
                    temp.kind = onechild;

                }
            }
        }
        else if (auto p = std::dynamic_pointer_cast<Perm>(exp))
        {
            //Impossible
            std::cerr << "Error: shouldn't be jump to perm\n";
        }
    }

    void Esat(std::shared_ptr<Output> program){
        //auto esat = ESaturation(program, repeat);
        //esat.saturate();
        std::unordered_set<ModeType> temp;

        DP_saturation(program->exp, temp);

    }

    

    void DP(std::shared_ptr<Exp> exp){
        if (auto i = std::dynamic_pointer_cast<Input>(exp))
        {
            //if the layout has LKC or KLC, the list has one layout, namely itself

            each layout has num_perms == 0 if there is a layout 
            
            and other has 1 perm

            return;
        }

    }
    //void DP(std::shared_ptr<Exp> exp){
    //    //2 lists: up_possible(by z3) and current_all nodes[build in DP], and others[placeholder]
    //    //current has node that must have perm upwards

    //    if (auto i = std::dynamic_pointer_cast<Input>(exp))
    //    {
    //        //if the layout has LKC or KLC, the list has one layout, namely itself
    //        each layout has num_perms == 0 if there is a layout 
    //        
    //        and other has 1 perm

    //        return;
    //    }
    //    else if (auto c = std::dynamic_pointer_cast<Contract>(exp))
    //    {
    //        DP l child and r child

    //        check l and r children's list
    //        l not empty   r not empty,  has same K.  
    //        1          1                     1           perms = l + r [if corresponding layout is in list, add to that layout pointing to corresponding l and r, new nodes are placed in current, othrewise produce a new layout, ]
    //        0          0                     ?           perms = l + r +2 [all nodes and others + l + r +2, pointing to others]
    //        0          1                     ?           perms = l + r +1 [generating nodes suitable for r, that node has l + r + 1, if alaready exising, add to exising up_possible]
    //        1          0                     ?            perms = l + r +1 [generating nodes suitable for l, that node has l + r + 1, if alaready exising, add to exising up_possible]
    //        1          1                     0            perms = l + r +1 [generating nodes suitable for l and r, that node has l + r + 1, if alaready exising, add to exising up_possible]

    //        //list has node M N K C
    //        a b -> c: m continu same? n continue same? k continue same?

    //        
    //    }

    //}

    /***********Print functions for debugging*************** */ 
    void time_print(float estimated_time, float real_time){
            std::cout << ", estimated time: " <<estimated_time<<  ", real time: " << real_time << " ms";
    }
    void Perm::print(const std::string &prefix, bool is_left) const
    {

        std::cout << prefix;
        std::cout << (is_left ? "├── " : "└── ");
        std::cout << "["<<name<<"] ";
        for (const auto &sub : get_subscripts())
        {
            std::cout << sub;
        }
        if (profiled)
        {
            time_print(estimated_time_ms, time_ms);
            std::cout << work_ptr;
        }
        
        std::cout << std::endl;

        in_exp->print(prefix + (is_left ? "│   " : "    "), true);
    }

    void Contract::print(const std::string &prefix, bool is_left) const
    {
        std::cout << prefix;
        std::cout << (is_left ? "├── " : "└── ");
        std::cout << "["<<name<<"] ";
        for (const auto &name : get_subscripts())
        {
            std::cout << name;
        }
        if (profiled)
        {
            time_print(estimated_time_ms, time_ms);
            std::cout << work_ptr;
        }
        std::cout << ", ari_int: " << ari_int;
        std::cout << std::endl;

        in_exp1->print(prefix + (is_left ? "│   " : "    "), true);
        in_exp2->print(prefix + (is_left ? "│   " : "    "), false);
    }

    void Input::print(const std::string &prefix, bool is_left) const
    {
        std::cout << prefix;
        std::cout << (is_left ? "├── " : "└── ");
        std::cout << "["<<name<<"] ";
        for (const auto &name : get_subscripts())
        {
            std::cout << name;
        }
        std::cout << ", Shape: (";
        for (const auto &extent : tensor->get_shape())
        {
            std::cout << extent << ",";
        }
        if (profiled)
        {
            time_print(estimated_time_ms, time_ms);
            std::cout << work_ptr;
        }
        std::cout << ")";
        std::cout << std::endl;
    }

    void Output::print() const
    {
        std::cout << "Context precision: " << (ctx.precision == FP32 ? "FP32" : "FP64") << std::endl;
        exp->print("", true);
        std::cout << std::endl;
    }

    void Output::expend() {
        expanded = true;
        auto new_exp = Expand_exp(exp);
        auto out_dims = exp->get_modes();

        if (out_dims != new_exp->get_modes())
        {
            new_exp = std::make_shared<Perm>(out_dims, ctx, new_exp);
        }

        exp = new_exp;
    }

} // namespace ETL

