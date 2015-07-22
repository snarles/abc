// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// I got this template from here: http://stackoverflow.com/a/18014655/1315767
template <typename WHAT>
class ListOf : public List {
public:
    template <typename T>
    ListOf( const T& x) : List(x){}

    WHAT operator[](int i){ return as<WHAT>( ( (List*)this)->operator[]( i) ) ; } 

} ;

// [[Rcpp::export]]


List FooList(NumericVector fi1, ListOf<NumericMatrix> Ct){

  List TempList(Ct.size());
  NumericMatrix ct(2,2);


  for(int i=0; i<Ct.size(); i++){
    ct = Ct[i] ;
    for (int j=0; j < ct.nrow(); j++) {
      for (int k=0; k < ct.ncol(); k++) {
        ct(j, k) *= fi1[i];  // Multiply each element of the matrix by the scalar in fi1
      }   
    }   
    TempList[i] = ct;                                                                                                                                                                                                                      
  }
   return TempList;
}