#include <iostream>

template <class T>
class Base {
	T first, second;
        public:
	Base(T a, T b){
                        first = a;
                        second = b;
                }
                inline T greater(){
                        return( first < second ? second : first);
                }
};

int main(){
  Base<double> b(42.42,88.7);
  Base<int> c(42, 77);
  std::cout << b.greater() << std::endl;
  auto compare = [] (double a, int b) {return a < b? a:b;};
  std::cout << compare(b.greater(), c.greater()) << std::endl;
}
