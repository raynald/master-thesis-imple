//=============================================================================
// File Name: WeightVector.cc
// implements the methods of the WeightVector class
//=============================================================================

#include "include/WeightVector.h"

void WeightVector::add(simple_sparse_vector& x, double s) {
    double pred = 0.0, norm_x = 0.0;
    for( simple_sparse_vector_iterator it = x.my_vec.begin(); 
            it != x.my_vec.end(); it++) {
        double val = (*it).second * s;
        norm_x += val*val;
        pred += 2.0*my_v[(*it).first]*val;
        my_v[(*it).first] += (val/my_a);
    }
    my_snorm += norm_x + my_a*pred;
}


// this += s*x
void WeightVector::add(WeightVector& x, double s) {
    x.make_my_a_one();
    my_snorm = 0.0;
    for (uint i=0; i<d; ++i) {
        my_v[i] *= my_a;
        my_v[i] += (x[i] * s);
        my_snorm += my_v[i]*my_v[i];
    }
    my_a = 1.0;
}

// special for AdaGrad
void WeightVector::sqr_add(WeightVector& x) {
    x.make_my_a_one();
    my_snorm = 0.0;
    for (uint i=0; i < d; ++i) {
        my_v[i] *= my_a;
        my_v[i] += (x[i] * x[i]);
        my_snorm += my_v[i] * my_v[i];
    }
    my_a = 1.0;
}

// special for AdaGrad
void WeightVector::pair_mul(WeightVector& x) {
    x.make_my_a_one();
    for (uint i=0; i < d; ++i) {
        if (my_v[i] != 0) my_v[i] /= sqrt(x[i]);
    }
}

void WeightVector::print(std::ostream& os) {

    for(uint i=0; i < d; ++i) {
        if (my_v[i] != 0.0)
            os << i << ":" << (my_v[i]*my_a) << " "; 
    }
    os << std::endl;
}

//--------------------------------------------------------------------------
double operator* (simple_sparse_vector& u, WeightVector& v) {
    double outcome = 0.0;
    for( simple_sparse_vector_iterator it = u.my_vec.begin(); 
            it != u.my_vec.end(); it++) {
        outcome += ((*it).second * v[(*it).first]);
    }
    return outcome;
}

//-----------------------------------------------------------------------------
double operator* (WeightVector& v, simple_sparse_vector& u) {
    return (u*v);
}


