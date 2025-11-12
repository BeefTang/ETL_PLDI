
struct {
   Modes up_L, up_C; 
   Modes down_M, down_N, down_C; 
   Modes bro_K;

   Modes get_modes(){
       Modes result;
       result.reserve(down_M.size() + down_N.size() + down_C.size());

       result.insert(result.end(), down_M.begin(), down_M.end());
       result.insert(result.end(), down_N.begin(), down_N.end());
       result.insert(result.end(), down_C.begin(), down_C.end());
       return result;
   }

   void extract_K(const std::set<ModeType> &K){
       auto order = get_modes();
       Modes result;
       result.reserve(K.size());  // optional

       for (int x : order) {
           if (K.count(x)) {  // check if element is in the set
               result.push_back(x);
           }
       }
   }


}
