#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>

#include "svm.h"

#ifdef NDEBUG
#include <assert.h>
#endif

#include <stdarg.h>
#if 1
void info(const char *fmt,...)
{
    va_list ap;
    va_start(ap,fmt);
    vprintf(fmt,ap);
    va_end(ap);
}
void info_flush()
{
    fflush(stdout);
}
#else
void info(char *fmt,...) {}
void info_flush() {}
#endif

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
inline double powi(double base, int times)
{
    double tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2){
	if(t%2==1) ret*=tmp;
	tmp = tmp * tmp;
    }
    return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//====CSOVO_WAP====
#include <algorithm>
void svm_train_csovo_wap(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_csovo_wap(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_csovo_wap(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_csovo_wap(FILE* fp, svm_model* model);
int svm_save_model_csovo_wap(FILE* fp, const svm_model *model);
//====CSOVA====
void svm_train_csova(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_csova(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_csova_cspcr_csosr(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_csova_cspcr_csosr(FILE* fp, svm_model* model);
int svm_save_model_csova_cspcr_csosr(FILE* fp, const svm_model *model);
//====CSPCR====
void svm_train_cspcr(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_cspcr_csosr(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_csova_cspcr_csosr(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_csova_cspcr_csosr(FILE* fp, svm_model* model);
int svm_save_model_csova_cspcr_csosr(FILE* fp, const svm_model *model);
//====CSOSR====
void svm_train_csosr(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_cspcr_csosr(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_csova_cspcr_csosr(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_csova_cspcr_csosr(FILE* fp, svm_model* model);
int svm_save_model_csova_cspcr_csosr(FILE* fp, const svm_model *model);
//====CSTREE/CSFT====
void svm_train_cstree_csft(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_cstree_csft(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_cstree_csft(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_cstree_csft(FILE* fp, svm_model* model);
int svm_save_model_cstree_csft(FILE* fp, const svm_model *model);
//====CSAPFT====
void svm_train_csapft(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_csapft(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_csapft(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_csapft(FILE* fp, svm_model* model);
int svm_save_model_csapft(FILE* fp, const svm_model *model);
//====CSSECOC====
void svm_train_cssecoc(const svm_problem *prob, const svm_parameter *param, svm_model *model);
void svm_predict_values_cssecoc(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict_cssecoc(const svm_model *model, const svm_node *x);
svm_model *svm_load_model_cssecoc(FILE* fp, svm_model* model);
int svm_save_model_cssecoc(FILE* fp, const svm_model *model);

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
    Cache(int l,long int size);
    ~Cache();

    // request data [0,len)
    // return some position p where [p,len) need to be filled
    // (p >= len if nothing needs to be filled)
    int get_data(const int index, Qfloat **data, int len);
    void swap_index(int i, int j);        // future_option
private:
    int l;
    long int size;
    struct head_t
    {
	head_t *prev, *next;        // a cicular list
	Qfloat *data;
	int len;            // data[0,len) is cached in this entry
    };
  
    head_t *head;
    head_t lru_head;
    void lru_delete(head_t *h);
    void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
    head = (head_t *)calloc(l,sizeof(head_t));    // initialized to 0
    size /= sizeof(Qfloat);
    size -= l * sizeof(head_t) / sizeof(Qfloat);
    size = max(size, (long int) 2*l);     // cache must be large enough for two columns
    lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
    for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
	free(h->data);
    free(head);
}

void Cache::lru_delete(head_t *h)
{
    // delete from current location
    h->prev->next = h->next;
    h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
    // insert to last position
    h->next = &lru_head;
    h->prev = lru_head.prev;
    h->prev->next = h;
    h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
    head_t *h = &head[index];
    if(h->len) lru_delete(h);
    int more = len - h->len;
  
    if(more > 0){
	// free old space
	while(size < more){
	    head_t *old = lru_head.next;
	    lru_delete(old);
	    free(old->data);
	    size += old->len;
	    old->data = 0;
	    old->len = 0;
	}
    
	// allocate new space
	h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
	size -= more;
	swap(h->len,len);
    }
  
    lru_insert(h);
    *data = h->data;
    return len;
}

void Cache::swap_index(int i, int j)
{
    if(i==j) return;
  
    if(head[i].len) lru_delete(&head[i]);
    if(head[j].len) lru_delete(&head[j]);
    swap(head[i].data,head[j].data);
    swap(head[i].len,head[j].len);
    if(head[i].len) lru_insert(&head[i]);
    if(head[j].len) lru_insert(&head[j]);
  
    if(i>j) swap(i,j);
    for(head_t *h = lru_head.next; h!=&lru_head; h=h->next){
	if(h->len > i){
	    if(h->len > j)
		swap(h->data[i],h->data[j]);
	    else{
		// give up
		lru_delete(h);
		free(h->data);
		size += h->len;
		h->data = 0;
		h->len = 0;
	    }
	}
    }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix 
{
public:
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual Qfloat *get_QD() const = 0;
    virtual void swap_index(int i, int j) const = 0;
    virtual ~QMatrix() {}
};

class Kernel: public QMatrix 
{
public:
    Kernel(int l, svm_node * const * x, const svm_parameter& param);
    virtual ~Kernel();

    static double k_function(const svm_node *x, const svm_node *y,
			     const svm_parameter& param);
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual Qfloat *get_QD() const = 0;
    virtual void swap_index(int i, int j) const   // no so const...
	{
	    swap(x[i],x[j]);
	    if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

    double (Kernel::*kernel_function)(int i, int j) const;

private:
    const svm_node **x;
    double *x_square;

    // svm_parameter
    const int kernel_type;
    const int degree;
    const double gamma;
    const double coef0;

    static double dot(const svm_node *px, const svm_node *py);
    static double dist_1(const svm_node * px, const svm_node * py);
    static double dist_2_sqr(const svm_node * px, const svm_node * py);
    
    inline double dist_2_sqr(int i, int j) const {
	double sum = x_square[i]+x_square[j]-2*dot(x[i],x[j]);
	return (sum > 0.0 ? sum : 0.0);
    }
    double kernel_linear(int i, int j) const {
	return dot(x[i],x[j]);
    }
    double kernel_poly(int i, int j) const {
	return powi(gamma*dot(x[i],x[j])+coef0,degree);
    }
    double kernel_rbf(int i, int j) const {
	return exp(-gamma*dist_2_sqr(i, j));
    }
    double kernel_sigmoid(int i, int j) const {
	return tanh(gamma*dot(x[i],x[j])+coef0);
    }
    double kernel_stump(int i, int j) const {
	return -dist_1(x[i], x[j])+coef0;
    }
    double kernel_perc(int i, int j) const {
	return -sqrt(dist_2_sqr(i, j))+coef0;
    }      
    double kernel_laplace(int i, int j) const {
	return exp(-gamma*dist_1(x[i], x[j]));
    }
    double kernel_expo(int i, int j) const {
	return exp(-gamma*sqrt(dist_2_sqr(i, j)));
    }
    double kernel_precomputed(int i, int j) const {
	return x[i][(int)(x[j][0].value)].value;
    }
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
    :kernel_type(param.kernel_type), degree(param.degree),
     gamma(param.gamma), coef0(param.coef0)
{
    switch(kernel_type){
    case LINEAR:
	kernel_function = &Kernel::kernel_linear;
	break;
    case POLY:
	kernel_function = &Kernel::kernel_poly;
	break;
    case RBF:
	kernel_function = &Kernel::kernel_rbf;
	break;
    case SIGMOID:
	kernel_function = &Kernel::kernel_sigmoid;
	break;
    case PRECOMPUTED:
	kernel_function = &Kernel::kernel_precomputed;
	break;
    case STUMP:
	kernel_function = &Kernel::kernel_stump;
	break;
    case PERC:
	kernel_function = &Kernel::kernel_perc;
	break;
    case LAPLACE:
	kernel_function = &Kernel::kernel_laplace;
	break;
    case EXPO:
	kernel_function = &Kernel::kernel_expo;
	break;
    }
  
    clone(x,x_,l);
  
    if(kernel_type == RBF || kernel_type == PERC || kernel_type == EXPO)
    {
	x_square = new double[l];
	for(int i=0;i<l;i++)
	    x_square[i] = dot(x[i],x[i]);
    }
    else
	x_square = 0;
}

Kernel::~Kernel()
{
    delete[] x;
    delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
	if(px->index == py->index)
        {
	    sum += px->value * py->value;
	    ++px;
	    ++py;
        }
	else
        {
	    if(px->index > py->index)
		++py;
	    else
		++px;
        }                       
    }
    return sum;
}

double Kernel::dist_1(const svm_node * px, const svm_node * py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
	if(px->index == py->index)
        {
	    sum += fabs(px->value - py->value);
	    ++px;
	    ++py;
        }
	else
        {
	    if(px->index > py->index)
            {
		sum += fabs(py->value);
		++py;
            }
	    else
            {
		sum += fabs(px->value);
		++px;
            }
        }
    
    }
    while(px->index != -1)
    {
	sum += fabs(px->value);
	++px;
    }
    while (py->index != -1)
    {
	sum += fabs (py->value);
	++py;
    }
    return sum;
}

double Kernel::dist_2_sqr(const svm_node * px, const svm_node * py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            double d = px->value - py->value;
            sum += d * d;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
            {
                sum += py->value * py->value;
                ++py;
            }
            else
            {
                sum += px->value * px->value;
                ++px;
            }
        }
    
    }
    while(px->index != -1)
    {
        sum += px->value * px->value;
        ++px;
    }
    while (py->index != -1)
    {
        sum += py->value * py->value;
        ++py;
    }

    return (sum > 0.0 ? sum : 0.0);
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
                          const svm_parameter& param)
{
    switch(param.kernel_type){
    case LINEAR:
        return dot(x,y);
    case POLY:
        return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
    case RBF:
        return exp(-param.gamma*dist_2_sqr(x, y));
    case SIGMOID:
        return tanh(param.gamma*dot(x,y)+param.coef0);
    case STUMP:
        return -dist_1(x, y) + param.coef0;
    case PERC:
        return -sqrt(dist_2_sqr(x, y)) + param.coef0;
    case LAPLACE:
        return exp(-param.gamma*dist_1(x, y));
    case EXPO:
        return exp(-param.gamma*sqrt(dist_2_sqr(x, y)));
    case PRECOMPUTED:  //x: test (validation), y: SV
        return x[(int)(y->value)].value;
    default:
        return 0; /* Unreachable */
    }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//      min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//              y^T \alpha = \delta
//              y_i = +1 or -1
//              0 <= alpha_i <= Cp for y_i = 1
//              0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//      Q, p, y, Cp, Cn, and an initial feasible point \alpha
//      l is the size of vectors and matrices
//      eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
    Solver() {};
    virtual ~Solver() {};

    struct SolutionInfo {
        double obj;
        double rho;
        double upper_bound_p;
        double upper_bound_n;
        double r;   // for Solver_NU
    };

    /** the solver with different C_i */
    void Solve(int l, const QMatrix& Q, const double *b_, const schar *y_,
               double *alpha_, const double *C_, double eps,
               SolutionInfo* si, int shrinking);

    void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
               double *alpha_, double Cp, double Cn, double eps,
               SolutionInfo* si, int shrinking);
protected:
    int active_size;
    schar *y;
    double *G;            // gradient of objective function
    /** the enum is removed to allow examples to be both at
        the upper bound and at the lower bound (if C = 0)
        enum { LOWER_BOUND, UPPER_BOUND, FREE };
    */
    static const char FREE = 0;
    static const char LOWER_BOUND = 1; // bit 1
    static const char UPPER_BOUND = 2; // bit 2
    char *alpha_status;   // LOWER_BOUND, UPPER_BOUND, FREE
    double *alpha;
    const QMatrix *Q;
    const Qfloat *QD;
    double eps;
    double *C;
    double *p;
    int *active_set;
    double *G_bar;                // gradient, if we treat free variables as 0
    int l;
    bool unshrinked;      // XXX

    double get_C(int i){
	return C[i];
    }
    /** the following modification is for maintaining the alpha_status */
    void update_alpha_status(int i){
	alpha_status[i] = FREE;
	if (alpha[i] <= 0)
	    alpha_status[i] |= LOWER_BOUND;
	if (alpha[i] >= get_C(i))
	    alpha_status[i] |= UPPER_BOUND;
    }
    bool is_upper_bound(int i) { return alpha_status[i] & UPPER_BOUND; }
    bool is_lower_bound(int i) { return alpha_status[i] & LOWER_BOUND; }
    bool is_free(int i) { return alpha_status[i] == FREE; }
    void swap_index(int i, int j);
    void reconstruct_gradient();
    virtual int select_working_set(int &i, int &j);
    virtual double calculate_rho();
    virtual void do_shrinking();
private:
    bool be_shrunken(int i, double Gmax1, double Gmax2);  
};

void Solver::swap_index(int i, int j)
{
    Q->swap_index(i,j);
    swap(y[i],y[j]);
    swap(G[i],G[j]);
    swap(alpha_status[i],alpha_status[j]);
    swap(alpha[i],alpha[j]);
    swap(p[i],p[j]);
    swap(active_set[i],active_set[j]);
    swap(G_bar[i],G_bar[j]);
    swap(C[i], C[j]);
}

void Solver::reconstruct_gradient()
{
    // reconstruct inactive elements of G from G_bar and free variables

    if(active_size == l) return;

    int i;
    for(i=active_size;i<l;i++)
        G[i] = G_bar[i] + p[i];
        
    for(i=0;i<active_size;i++)
        if(is_free(i))
        {
            const Qfloat *Q_i = Q->get_Q(i,l);
            double alpha_i = alpha[i];
            for(int j=active_size;j<l;j++)
                G[j] += alpha_i * Q_i[j];
        }
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
                   double *alpha_, const double* C_, double eps,
                   SolutionInfo* si, int shrinking)
{
    this->l = l;
    this->Q = &Q;
    QD=Q.get_QD();
    clone(p, p_,l);
    clone(y, y_,l);
    clone(alpha,alpha_,l);
    clone(C, C_, l);
    this->eps = eps;
    unshrinked = false;

    // initialize alpha_status
    {
        alpha_status = new char[l];
        for(int i=0;i<l;i++)
            update_alpha_status(i);
    }

    // initialize active set (for shrinking)
    {
        active_set = new int[l];
        for(int i=0;i<l;i++)
            active_set[i] = i;
        active_size = l;
    }

    // initialize gradient
    {
        G = new double[l];
        G_bar = new double[l];
        int i;
        for(i=0;i<l;i++)
        {
            G[i] = p[i];
            G_bar[i] = 0;
        }
        for(i=0;i<l;i++)
            if(!is_lower_bound(i))
            {
                const Qfloat *Q_i = Q.get_Q(i,l);
                double alpha_i = alpha[i];
                int j;
                for(j=0;j<l;j++)
                    G[j] += alpha_i*Q_i[j];
                if(is_upper_bound(i))
                    for(j=0;j<l;j++)
                        G_bar[j] += get_C(i) * Q_i[j];
            }
    }

    // optimization step

    int iter = 0;
    int counter = min(l,1000)+1;

    while(1)
    {
        // show progress and do shrinking

        if(--counter == 0)
        {
            counter = min(l,1000);
            if(shrinking) do_shrinking();
            info("."); info_flush();
        }

        int i,j;
        if(select_working_set(i,j)!=0)
        {
            // reconstruct the whole gradient
            reconstruct_gradient();
            // reset active set size and check
            active_size = l;
            info("*"); info_flush();
            if(select_working_set(i,j)!=0)
                break;
            else
                counter = 1;        // do shrinking next iteration
        }
        ++iter;

        // update alpha[i] and alpha[j], handle bounds carefully
                
        const Qfloat *Q_i = Q.get_Q(i,active_size);
        const Qfloat *Q_j = Q.get_Q(j,active_size);

        double C_i = get_C(i);
        double C_j = get_C(j);

        double old_alpha_i = alpha[i];
        double old_alpha_j = alpha[j];

        if(y[i]!=y[j])
        {
            double quad_coef = Q_i[i]+Q_j[j]+2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            double delta = (-G[i]-G[j])/quad_coef;
            double diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
                        
            if(diff > 0)
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = diff;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = -diff;
                }
            }
            if(diff > C_i - C_j)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = C_i - diff;
                }
            }
            else
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = C_j + diff;
                }
            }
        }
        else
        {
            double quad_coef = Q_i[i]+Q_j[j]-2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            double delta = (G[i]-G[j])/quad_coef;
            double sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;

            if(sum > C_i)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = sum - C_i;
                }
            }
            else
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = sum;
                }
            }
            if(sum > C_j)
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = sum - C_j;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = sum;
                }
            }
        }

        // update G

        double delta_alpha_i = alpha[i] - old_alpha_i;
        double delta_alpha_j = alpha[j] - old_alpha_j;
                
        for(int k=0;k<active_size;k++)
        {
            G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
        }

        // update alpha_status and G_bar

        {
            bool ui = is_upper_bound(i);
            bool uj = is_upper_bound(j);
            update_alpha_status(i);
            update_alpha_status(j);
            int k;
            if(ui != is_upper_bound(i))
            {
                Q_i = Q.get_Q(i,l);
                if(ui)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_i * Q_i[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_i * Q_i[k];
            }

            if(uj != is_upper_bound(j))
            {
                Q_j = Q.get_Q(j,l);
                if(uj)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_j * Q_j[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_j * Q_j[k];
            }
        }
    }

    // calculate rho

    si->rho = calculate_rho();

    // calculate objective value
    {
        double v = 0;
        int i;
        for(i=0;i<l;i++)
            v += alpha[i] * (G[i] + p[i]);

        si->obj = v/2;
    }

    // put back the solution
    {
        for(int i=0;i<l;i++)
            alpha_[active_set[i]] = alpha[i];
    }

    // juggle everything back
    /*{
      for(int i=0;i<l;i++)
      while(active_set[i] != i)
      swap_index(i,active_set[i]);
      // or Q.swap_index(i,active_set[i]);
      }*/

    info("\noptimization finished, #iter = %d\n",iter);

    delete[] p;
    delete[] y;
    delete[] C;
    delete[] alpha;
    delete[] alpha_status;
    delete[] active_set;
    delete[] G;
    delete[] G_bar;
}

void Solver::Solve(int l, const QMatrix& Q, const double *b_, const schar *y_,
                   double *alpha_, double Cp, double Cn, double eps,
                   SolutionInfo* si, int shrinking)
{
    double* C_ = new double[l];
    for (int i = 0; i < l; ++i)
        C_[i] = (y_[i] > 0 ? Cp : Cn);
    Solve(l, Q, b_, y_, alpha_, C_, eps, si, shrinking);
    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;
    delete[] C_;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
    // return i,j such that
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
        
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)        
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmax)
                {
                    Gmax = -G[t];
                    Gmax_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmax)
                {
                    Gmax = G[t];
                    Gmax_idx = t;
                }
        }

    int i = Gmax_idx;
    const Qfloat *Q_i = NULL;
    if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
        Q_i = Q->get_Q(i,active_size);

    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j))
            {
                double grad_diff=Gmax+G[j];
                if (G[j] >= Gmax2)
                    Gmax2 = G[j];
                if (grad_diff > 0)
                {
                    double obj_diff; 
                    double quad_coef=Q_i[i]+QD[j]-2*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                double grad_diff= Gmax-G[j];
                if (-G[j] >= Gmax2)
                    Gmax2 = -G[j];
                if (grad_diff > 0)
                {
                    double obj_diff; 
                    double quad_coef=Q_i[i]+QD[j]+2*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    if(Gmax+Gmax2 < eps)
        return 1;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return 0;
}

bool Solver::be_shrunken(int i, double Gmax1, double Gmax2)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else
            return(-G[i] > Gmax2);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else      
            return(G[i] > Gmax1);
    }
    else
        return(false);
}

void Solver::do_shrinking()
{
    int i;
    double Gmax1 = -INF;          // max { -y_i * grad(f)_i | i in I_up(\alpha) }
    double Gmax2 = -INF;          // max { y_i * grad(f)_i | i in I_low(\alpha) }

    // find maximal violating pair first
    for(i=0;i<active_size;i++)
    {
        if(y[i]==+1)      
        {
            if(!is_upper_bound(i))        
            {
                if(-G[i] >= Gmax1)
                    Gmax1 = -G[i];
            }
            if(!is_lower_bound(i))        
            {
                if(G[i] >= Gmax2)
                    Gmax2 = G[i];
            }
        }
        else      
        {
            if(!is_upper_bound(i))        
            {
                if(-G[i] >= Gmax2)
                    Gmax2 = -G[i];
            }
            if(!is_lower_bound(i))        
            {
                if(G[i] >= Gmax1)
                    Gmax1 = G[i];
            }
        }
    }

    // shrink

    for(i=0;i<active_size;i++)
        if (be_shrunken(i, Gmax1, Gmax2))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunken(active_size, Gmax1, Gmax2))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }

    // unshrink, check all variables again before final iterations

    if(unshrinked || Gmax1 + Gmax2 > eps*10) return;
        
    unshrinked = true;
    reconstruct_gradient();

    for(i=l-1;i>=active_size;i--)
        if (!be_shrunken(i, Gmax1, Gmax2))
        {
            while (active_size < i)
            {
                if (be_shrunken(active_size, Gmax1, Gmax2))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size++;
            }
            active_size++;
        }
}

double Solver::calculate_rho()
{
    double r;
    int nr_free = 0;
    double ub = INF, lb = -INF, sum_free = 0;
    for(int i=0;i<active_size;i++)
    {
        double yG = y[i]*G[i];

        if(is_upper_bound(i))
        {
            if(y[i]==-1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else if(is_lower_bound(i))
        {
            if(y[i]==+1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else
        {
            ++nr_free;
            sum_free += yG;
        }
    }

    if(nr_free>0)
        r = sum_free/nr_free;
    else
        r = (ub+lb)/2;

    return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
    Solver_NU() {}
    void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
               double *alpha, double Cp, double Cn, double eps,
               SolutionInfo* si, int shrinking)
        {
            this->si = si;
            Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
        }
private:
    SolutionInfo *si;
    int select_working_set(int &i, int &j);
    double calculate_rho();
    bool be_shrunken(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
    void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
    // return i,j such that y_i = y_j and
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

    double Gmaxp = -INF;
    double Gmaxp2 = -INF;
    int Gmaxp_idx = -1;

    double Gmaxn = -INF;
    double Gmaxn2 = -INF;
    int Gmaxn_idx = -1;

    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmaxp)
                {
                    Gmaxp = -G[t];
                    Gmaxp_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmaxn)
                {
                    Gmaxn = G[t];
                    Gmaxn_idx = t;
                }
        }

    int ip = Gmaxp_idx;
    int in = Gmaxn_idx;
    const Qfloat *Q_ip = NULL;
    const Qfloat *Q_in = NULL;
    if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
        Q_ip = Q->get_Q(ip,active_size);
    if(in != -1)
        Q_in = Q->get_Q(in,active_size);

    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j))       
            {
                double grad_diff=Gmaxp+G[j];
                if (G[j] >= Gmaxp2)
                    Gmaxp2 = G[j];
                if (grad_diff > 0)
                {
                    double obj_diff; 
                    double quad_coef = Q_ip[ip]+QD[j]-2*Q_ip[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                double grad_diff=Gmaxn-G[j];
                if (-G[j] >= Gmaxn2)
                    Gmaxn2 = -G[j];
                if (grad_diff > 0)
                {
                    double obj_diff; 
                    double quad_coef = Q_in[in]+QD[j]-2*Q_in[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
        return 1;

    if (y[Gmin_idx] == +1)
        out_i = Gmaxp_idx;
    else
        out_i = Gmaxn_idx;
    out_j = Gmin_idx;

    return 0;
}

bool Solver_NU::be_shrunken(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else      
            return(-G[i] > Gmax4);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else      
            return(G[i] > Gmax3);
    }
    else
        return(false);
}

void Solver_NU::do_shrinking()
{
    double Gmax1 = -INF;  // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
    double Gmax2 = -INF;  // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
    double Gmax3 = -INF;  // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
    double Gmax4 = -INF;  // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

    // find maximal violating pair first
    int i;
    for(i=0;i<active_size;i++)
    {
        if(!is_upper_bound(i))
        {
            if(y[i]==+1)
            {
                if(-G[i] > Gmax1) Gmax1 = -G[i];
            }
            else  if(-G[i] > Gmax4) Gmax4 = -G[i];
        }
        if(!is_lower_bound(i))
        {
            if(y[i]==+1)
            {   
                if(G[i] > Gmax2) Gmax2 = G[i];
            }
            else  if(G[i] > Gmax3) Gmax3 = G[i];
        }
    }

    // shrinking

    for(i=0;i<active_size;i++)
        if (be_shrunken(i, Gmax1, Gmax2, Gmax3, Gmax4))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunken(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }

    // unshrink, check all variables again before final iterations

    if(unshrinked || max(Gmax1+Gmax2,Gmax3+Gmax4) > eps*10) return;
        
    unshrinked = true;
    reconstruct_gradient();

    for(i=l-1;i>=active_size;i--)
        if (!be_shrunken(i, Gmax1, Gmax2, Gmax3, Gmax4))
        {
            while (active_size < i)
            {
                if (be_shrunken(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size++;
            }
            active_size++;
        }
}

double Solver_NU::calculate_rho()
{
    int nr_free1 = 0,nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

    for(int i=0;i<active_size;i++)
    {
        if(y[i]==+1)
        {
            if(is_upper_bound(i))
                lb1 = max(lb1,G[i]);
            else if(is_lower_bound(i))
                ub1 = min(ub1,G[i]);
            else
            {
                ++nr_free1;
                sum_free1 += G[i];
            }
        }
        else
        {
            if(is_upper_bound(i))
                lb2 = max(lb2,G[i]);
            else if(is_lower_bound(i))
                ub2 = min(ub2,G[i]);
            else
            {
                ++nr_free2;
                sum_free2 += G[i];
            }
        }
    }

    double r1,r2;
    if(nr_free1 > 0)
        r1 = sum_free1/nr_free1;
    else
        r1 = (ub1+lb1)/2;
        
    if(nr_free2 > 0)
        r2 = sum_free2/nr_free2;
    else
        r2 = (ub2+lb2)/2;
        
    si->r = (r1+r2)/2;
    return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
    SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
        :Kernel(prob.l, prob.x, param)
        {
            clone(y,y_,prob.l);
            cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
            QD = new Qfloat[prob.l];
            for(int i=0;i<prob.l;i++)
                QD[i]= (Qfloat)(this->*kernel_function)(i,i);
        }
        
    Qfloat *get_Q(int i, int len) const
        {
            Qfloat *data;
            int start;
            if((start = cache->get_data(i,&data,len)) < len)
            {
                int j;
#pragma omp parallel for private(j) 
                for(j=start;j<len;j++)
                    data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
            }
            return data;
        }

    Qfloat *get_QD() const
        {
            return QD;
        }

    void swap_index(int i, int j) const
        {
            cache->swap_index(i,j);
            Kernel::swap_index(i,j);
            swap(y[i],y[j]);
            swap(QD[i],QD[j]);
        }

    ~SVC_Q()
        {
            delete[] y;
            delete cache;
            delete[] QD;
        }
private:
    schar *y;
    Cache *cache;
    Qfloat *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
    ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
        :Kernel(prob.l, prob.x, param)
        {
            cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
            QD = new Qfloat[prob.l];
            for(int i=0;i<prob.l;i++)
                QD[i]= (Qfloat)(this->*kernel_function)(i,i);
        }
        
    Qfloat *get_Q(int i, int len) const
        {
            Qfloat *data;
            int start;
            if((start = cache->get_data(i,&data,len)) < len)
            {
                for(int j=start;j<len;j++)
                    data[j] = (Qfloat)(this->*kernel_function)(i,j);
            }
            return data;
        }

    Qfloat *get_QD() const
        {
            return QD;
        }

    void swap_index(int i, int j) const
        {
            cache->swap_index(i,j);
            Kernel::swap_index(i,j);
            swap(QD[i],QD[j]);
        }

    ~ONE_CLASS_Q()
        {
            delete cache;
            delete[] QD;
        }
private:
    Cache *cache;
    Qfloat *QD;
};

class SVR_Q: public Kernel
{ 
public:
    SVR_Q(const svm_problem& prob, const svm_parameter& param)
        :Kernel(prob.l, prob.x, param)
        {
            l = prob.l;
            cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
            QD = new Qfloat[2*l];
            sign = new schar[2*l];
            index = new int[2*l];
            for(int k=0;k<l;k++)
            {
                sign[k] = 1;
                sign[k+l] = -1;
                index[k] = k;
                index[k+l] = k;
                QD[k]= (Qfloat)(this->*kernel_function)(k,k);
                QD[k+l]=QD[k];
            }
            buffer[0] = new Qfloat[2*l];
            buffer[1] = new Qfloat[2*l];
            next_buffer = 0;
        }

    void swap_index(int i, int j) const
        {
            swap(sign[i],sign[j]);
            swap(index[i],index[j]);
            swap(QD[i],QD[j]);
        }
        
    Qfloat *get_Q(int i, int len) const
        {
            Qfloat *data;
            int real_i = index[i];
            if(cache->get_data(real_i,&data,l) < l)
            {
                for(int j=0;j<l;j++)
                    data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
            }

            // reorder and copy
            Qfloat *buf = buffer[next_buffer];
            next_buffer = 1 - next_buffer;
            schar si = sign[i];
            for(int j=0;j<len;j++)
                buf[j] = si * sign[j] * data[index[j]];
            return buf;
        }

    Qfloat *get_QD() const
        {
            return QD;
        }

    ~SVR_Q()
        {
            delete cache;
            delete[] sign;
            delete[] index;
            delete[] buffer[0];
            delete[] buffer[1];
            delete[] QD;
        }
private:
    int l;
    Cache *cache;
    schar *sign;
    int *index;
    mutable int next_buffer;
    Qfloat *buffer[2];
    Qfloat *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
    const svm_problem *prob, const svm_parameter* param,
    double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
    int l = prob->l;
    double *minus_ones = new double[l];
    schar *y = new schar[l];

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -1;
        if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
    }

    Solver s;
    s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
            alpha, Cp, Cn, param->eps, si, param->shrinking);

    double sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    if (Cp==Cn)
        info("nu = %f\n", sum_alpha/(Cp*prob->l));

    for(i=0;i<l;i++)
        alpha[i] *= y[i];

    delete[] minus_ones;
    delete[] y;
}

static void solve_cost_svc(
    const svm_problem *prob, const svm_parameter* param,
    double *alpha, Solver::SolutionInfo* si, double* C)
{
    int l = prob->l;
    double *minus_ones = new double[l];
    schar *y = new schar[l];

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -1;
        if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
    }

    Solver s;
    s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
            alpha, C, param->eps, si, param->shrinking);

    double sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    for(i=0;i<l;i++)
        alpha[i] *= y[i];

    delete[] minus_ones;
    delete[] y;
}

static void solve_cost_osr(
    const svm_problem *prob, const svm_parameter* param,
    double *alpha, Solver::SolutionInfo* si, double* RHS, double* C)
{
    int l = prob->l;
    double *minus_ones = new double[l];
    schar *y = new schar[l];

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -RHS[i];
        if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
    }

    Solver s;
    s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
            alpha, C, param->eps, si, param->shrinking);

    double sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    for(i=0;i<l;i++)
        alpha[i] *= y[i];

    delete[] minus_ones;
    delete[] y;
}

static void solve_nu_svc(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int i;
    int l = prob->l;
    double nu = param->nu;

    schar *y = new schar[l];

    for(i=0;i<l;i++)
        if(prob->y[i]>0)
            y[i] = +1;
        else
            y[i] = -1;

    double sum_pos = nu*l/2;
    double sum_neg = nu*l/2;

    for(i=0;i<l;i++)
        if(y[i] == +1)
        {
            alpha[i] = min(1.0,sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = min(1.0,sum_neg);
            sum_neg -= alpha[i];
        }

    double *zeros = new double[l];

    for(i=0;i<l;i++)
        zeros[i] = 0;

    Solver_NU s;
    s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
            alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
    double r = si->r;

    info("C = %f\n",1/r);

    for(i=0;i<l;i++)
        alpha[i] *= y[i]/r;

    si->rho /= r;
    si->obj /= (r*r);
    si->upper_bound_p = 1/r;
    si->upper_bound_n = 1/r;

    delete[] y;
    delete[] zeros;
}

static void solve_one_class(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *zeros = new double[l];
    schar *ones = new schar[l];
    int i;

    int n = (int)(param->nu*prob->l);     // # of alpha's at upper bound

    for(i=0;i<n;i++)
        alpha[i] = 1;
    if(n<prob->l)
        alpha[n] = param->nu * prob->l - n;
    for(i=n+1;i<l;i++)
        alpha[i] = 0;

    for(i=0;i<l;i++)
    {
        zeros[i] = 0;
        ones[i] = 1;
    }

    Solver s;
    s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
            alpha, 1.0, 1.0, param->eps, si, param->shrinking);

    delete[] zeros;
    delete[] ones;
}

static void solve_epsilon_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    for(i=0;i<l;i++)
    {
        alpha2[i] = 0;
        linear_term[i] = param->p - prob->y[i];
        y[i] = 1;

        alpha2[i+l] = 0;
        linear_term[i+l] = param->p + prob->y[i];
        y[i+l] = -1;
    }

    Solver s;
    s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
            alpha2, param->C, param->C, param->eps, si, param->shrinking);

    double sum_alpha = 0;
    for(i=0;i<l;i++)
    {
        alpha[i] = alpha2[i] - alpha2[i+l];
        sum_alpha += fabs(alpha[i]);
    }
    info("nu = %f\n",sum_alpha/(param->C*l));

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

static void solve_nu_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double C = param->C;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    double sum = C * param->nu * l / 2;
    for(i=0;i<l;i++)
    {
        alpha2[i] = alpha2[i+l] = min(sum,C);
        sum -= alpha2[i];

        linear_term[i] = - prob->y[i];
        y[i] = 1;

        linear_term[i+l] = prob->y[i];
        y[i+l] = -1;
    }

    Solver_NU s;
    s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
            alpha2, C, C, param->eps, si, param->shrinking);

    info("epsilon = %f\n",-si->r);

    for(i=0;i<l;i++)
        alpha[i] = alpha2[i] - alpha2[i+l];

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

decision_function svm_train_one(
    const svm_problem *prob, const svm_parameter *param,
    double* RHS, double *C)
{
    double *alpha = Malloc(double,prob->l);
    Solver::SolutionInfo si;
    switch(param->svm_type){
    case CSOSR:
        solve_cost_osr(prob,param,alpha,&si,RHS,C);
        break;
    default:
        fprintf(stderr, "supported only for CSOSR\n");
        exit(1);
    }

    info("obj = %f, rho = %f\n",si.obj,si.rho);

    // output SVs

    int nSV = 0;
    int nBSV = 0;
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(fabs(alpha[i]) >= C[i])
                ++nBSV;
        }
    }

    info("nSV = %d, nBSV = %d\n",nSV,nBSV);

    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}

decision_function svm_train_one(
    const svm_problem *prob, const svm_parameter *param,
    double *C)
{
    double *alpha = Malloc(double,prob->l);
    Solver::SolutionInfo si;
    switch(param->svm_type){
    case CSOVO_SVC:
    case WAP_SVC:
    case CSOVA_SVC:
    case CSTREE_SVC:
    case CSFT_SVC:
    case CSSECOC_SVC:
    case CSAPFT_SVC:
        solve_cost_svc(prob,param,alpha,&si,C);
        break;
    default:
        fprintf(stderr, "weighted C supported only for CSOVO_SVC and WAP_SVC\n");
        exit(1);
    }

    info("obj = %f, rho = %f\n",si.obj,si.rho);

    // output SVs

    int nSV = 0;
    int nBSV = 0;
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(fabs(alpha[i]) >= C[i])
                ++nBSV;
        }
    }

    info("nSV = %d, nBSV = %d\n",nSV,nBSV);

    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}

decision_function svm_train_one(
    const svm_problem *prob, const svm_parameter *param,
    double Cp, double Cn)
{
    double *alpha = Malloc(double,prob->l);
    Solver::SolutionInfo si;
    switch(param->svm_type){
    case C_SVC:
        solve_c_svc(prob,param,alpha,&si,Cp,Cn);
        break;
    case NU_SVC:
        solve_nu_svc(prob,param,alpha,&si);
        break;
    case ONE_CLASS:
        solve_one_class(prob,param,alpha,&si);
        break;
    case EPSILON_SVR:
        solve_epsilon_svr(prob,param,alpha,&si);
        break;
    case NU_SVR:
        solve_nu_svr(prob,param,alpha,&si);
        break;
    }

    info("obj = %f, rho = %f\n",si.obj,si.rho);

    // output SVs

    int nSV = 0;
    int nBSV = 0;
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(prob->y[i] > 0)
            {
                if(fabs(alpha[i]) >= si.upper_bound_p)
                    ++nBSV;
            }
            else
            {
                if(fabs(alpha[i]) >= si.upper_bound_n)
                    ++nBSV;
            }
        }
    }

    info("nSV = %d, nBSV = %d\n",nSV,nBSV);

    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
void sigmoid_train(
    int l, const double *dec_values, const double *labels, 
    double& A, double& B)
{
    double prior1=0, prior0 = 0;
    int i;

    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;
        
    int max_iter=100;     // Maximal number of iterations
    double min_step=1e-10;        // Minimal step taken in line search
    double sigma=1e-3;    // For numerically strict PD of Hessian
    double eps=1e-5;
    double hiTarget=(prior1+1.0)/(prior1+2.0);
    double loTarget=1/(prior0+2.0);
    double *t=Malloc(double,l);
    double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    double newA,newB,newf,d1,d2;
    int iter; 
        
    // Initial Point and Initial Fun Value
    A=0.0; B=log((prior0+1.0)/(prior1+1.0));
    double fval = 0.0;

    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // Stopping Criteria
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;


        stepsize = 1;             // Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step)
        {
            info("Line search fails in two-class probability estimates\n");
            break;
        }
    }

    if (iter>=max_iter)
        info("Reaching maximal iterations in two-class probability estimates\n");
    free(t);
}

double sigmoid_predict(double decision_value, double A, double B)
{
    double fApB = decision_value*A+B;
    if (fApB >= 0)
        return exp(-fApB)/(1.0+exp(-fApB));
    else
        return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
void multiclass_probability(int k, double **r, double *p)
{
    int t,j;
    int iter = 0, max_iter=max(100,k);
    double **Q=Malloc(double *,k);
    double *Qp=Malloc(double,k);
    double pQp, eps=0.005/k;
        
    for (t=0;t<k;t++)
    {
        p[t]=1.0/k;  // Valid if k = 1
        Q[t]=Malloc(double,k);
        Q[t][t]=0;
        for (j=0;j<t;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=Q[j][t];
        }
        for (j=t+1;j<k;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=-r[j][t]*r[t][j];
        }
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0;
        for (t=0;t<k;t++)
        {
            Qp[t]=0;
            for (j=0;j<k;j++)
                Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        double max_error=0;
        for (t=0;t<k;t++)
        {
            double error=fabs(Qp[t]-pQp);
            if (error>max_error)
                max_error=error;
        }
        if (max_error<eps) break;
                
        for (t=0;t<k;t++)
        {
            double diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
            for (j=0;j<k;j++)
            {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    if (iter>=max_iter)
        info("Exceeds max_iter in multiclass_prob\n");
    for(t=0;t<k;t++) free(Q[t]);
    free(Q);
    free(Qp);
}

// Cross-validation decision values for probability estimates
void svm_binary_svc_probability(
    const svm_problem *prob, const svm_parameter *param,
    double Cp, double Cn, double& probA, double& probB)
{
    int i;
    int nr_fold = 5;
    int *perm = Malloc(int,prob->l);
    double *dec_values = Malloc(double,prob->l);

    // random shuffle
    for(i=0;i<prob->l;i++) perm[i]=i;
    for(i=0;i<prob->l;i++)
    {
        int j = i+rand()%(prob->l-i);
        swap(perm[i],perm[j]);
    }
    for(i=0;i<nr_fold;i++)
    {
        int begin = i*prob->l/nr_fold;
        int end = (i+1)*prob->l/nr_fold;
        int j,k;
        struct svm_problem subprob;

        subprob.l = prob->l-(end-begin);
        subprob.x = Malloc(struct svm_node*,subprob.l);
        subprob.y = Malloc(double,subprob.l);
                        
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<prob->l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        int p_count=0,n_count=0;
        for(j=0;j<k;j++)
            if(subprob.y[j]>0)
                p_count++;
            else
                n_count++;

        if(p_count==0 && n_count==0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 0;
        else if(p_count > 0 && n_count == 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 1;
        else if(p_count == 0 && n_count > 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = -1;
        else
        {
            svm_parameter subparam = *param;
            subparam.probability=0;
            subparam.C=1.0;
            subparam.nr_weight=2;
            subparam.weight_label = Malloc(int,2);
            subparam.weight = Malloc(double,2);
            subparam.weight_label[0]=+1;
            subparam.weight_label[1]=-1;
            subparam.weight[0]=Cp;
            subparam.weight[1]=Cn;
            struct svm_model *submodel = svm_train(&subprob,&subparam);
            for(j=begin;j<end;j++)
            {
                svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
                // ensure +1 -1 order; reason not using CV subroutine
                dec_values[perm[j]] *= submodel->label[0];
            }           
            svm_destroy_model(submodel);
            svm_destroy_param(&subparam);
        }
        free(subprob.x);
        free(subprob.y);
    }           
    sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
    free(dec_values);
    free(perm);
}

// Return parameter of a Laplace distribution 
double svm_svr_probability(
    const svm_problem *prob, const svm_parameter *param)
{
    int i;
    int nr_fold = 5;
    double *ymv = Malloc(double,prob->l);
    double mae = 0;

    svm_parameter newparam = *param;
    newparam.probability = 0;
    svm_cross_validation(prob,&newparam,nr_fold,ymv);
    for(i=0;i<prob->l;i++)
    {
        ymv[i]=prob->y[i]-ymv[i];
        mae += fabs(ymv[i]);
    }           
    mae /= prob->l;
    double std=sqrt(2*mae*mae);
    int count=0;
    mae=0;
    for(i=0;i<prob->l;i++)
        if (fabs(ymv[i]) > 5*std) 
            count=count+1;
        else 
            mae+=fabs(ymv[i]);
    mae /= (prob->l-count);
    info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
    free(ymv);
    return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = Malloc(int,max_nr_class);
    int *count = Malloc(int,max_nr_class);
    int *data_label = Malloc(int,l);      
    int i;

    for(i=0;i<l;i++)
    {
        int this_label = (int)prob->y[i];
        int j;
        for(j=0;j<nr_class;j++)
        {
            if(this_label == label[j])
            {
                ++count[j];
                break;
            }
        }
        data_label[i] = j;
        if(j == nr_class)
        {
            if(nr_class == max_nr_class)
            {
                max_nr_class *= 2;
                label = (int *)realloc(label,max_nr_class*sizeof(int));
                count = (int *)realloc(count,max_nr_class*sizeof(int));
            }
            label[nr_class] = this_label;
            count[nr_class] = 1;
            ++nr_class;
        }
    }

    int *start = Malloc(int,nr_class);
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];
    for(i=0;i<l;i++)
    {
        perm[start[data_label[i]]] = i;
        ++start[data_label[i]];
    }
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];

    *nr_class_ret = nr_class;
    *label_ret = label;
    *start_ret = start;
    *count_ret = count;
    free(data_label);
}

//
// Interface functions
//

svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
    svm_model *model = Malloc(svm_model,1);
    model->param = *param;
    model->free_sv = 0;   // XXX

    if(param->svm_type == ONE_CLASS ||
       param->svm_type == EPSILON_SVR ||
       param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;
        model->label = NULL;
        model->nSV = NULL;
        model->probA = NULL; model->probB = NULL;
        model->sv_coef = Malloc(double *,1);

        if(param->probability && 
           (param->svm_type == EPSILON_SVR ||
            param->svm_type == NU_SVR))
        {
            model->probA = Malloc(double,1);
            model->probA[0] = svm_svr_probability(prob,param);
        }

        decision_function f = svm_train_one(prob,param,(double)0,(double)0);
        model->rho = Malloc(double,1);
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        model->SV = Malloc(svm_node *,nSV);
        model->sv_coef[0] = Malloc(double,nSV);
        int j = 0;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = prob->x[i];
                model->sv_coef[0][j] = f.alpha[i];
                ++j;
            }             

        free(f.alpha);
    }
    else if (param->svm_type == NU_SVC || param->svm_type == C_SVC)
    {
        // NU_SVC classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = Malloc(int,l);

        // group training data of the same class
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);              
        svm_node **x = Malloc(svm_node *,l);
        int i;
        for(i=0;i<l;i++)
            x[i] = prob->x[perm[i]];

        // calculate weighted C

        double *weighted_C = Malloc(double, nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        for(i=0;i<param->nr_weight;i++)
        {       
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // train k*(k-1)/2 models
                
        bool *nonzero = Malloc(bool,l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
        decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

        double *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=Malloc(double,nr_class*(nr_class-1)/2);
            probB=Malloc(double,nr_class*(nr_class-1)/2);
        }

        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                svm_problem sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.x = Malloc(svm_node *,sub_prob.l);
                sub_prob.y = Malloc(double,sub_prob.l);
                int k;
                for(k=0;k<ci;k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.y[k] = +1;
                }
                for(k=0;k<cj;k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.y[ci+k] = -1;
                }

                if(param->probability)
                    svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

                f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
                for(k=0;k<ci;k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0;k<cj;k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.y);
                ++p;
            }

        // build output

        model->nr_class = nr_class;
                
        model->label = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            model->label[i] = label[i];
                
        model->rho = Malloc(double,nr_class*(nr_class-1)/2);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            model->rho[i] = f[i].rho;

        if(param->probability)
        {
            model->probA = Malloc(double,nr_class*(nr_class-1)/2);
            model->probB = Malloc(double,nr_class*(nr_class-1)/2);
            for(i=0;i<nr_class*(nr_class-1)/2;i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            model->probA=NULL;
            model->probB=NULL;
        }

        int total_sv = 0;
        int *nz_count = Malloc(int,nr_class);
        model->nSV = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
        {
            int nSV = 0;
            for(int j=0;j<count[i];j++)
                if(nonzero[start[i]+j])
                { 
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }
                
        info("Total nSV = %d\n",total_sv);

        model->l = total_sv;
        model->SV = Malloc(svm_node *,total_sv);
        p = 0;
        for(i=0;i<l;i++)
            if(nonzero[i]) model->SV[p++] = x[i];

        int *nz_start = Malloc(int,nr_class);
        nz_start[0] = 0;
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        model->sv_coef = Malloc(double *,nr_class-1);
        for(i=0;i<nr_class-1;i++)
            model->sv_coef[i] = Malloc(double,total_sv);

        p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];
                                
                int q = nz_start[i];
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }
                
        free(label);
        free(probA);
        free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(weighted_C);
        free(nonzero);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
    else if (param->svm_type == CSOVO_SVC || param->svm_type == WAP_SVC)
        svm_train_csovo_wap(prob, param, model);
    else if (param->svm_type == CSOVA_SVC)
        svm_train_csova(prob, param, model);
    else if (param->svm_type == CSPCR_ESVR)
        svm_train_cspcr(prob, param, model);
    else if (param->svm_type == CSOSR)
        svm_train_csosr(prob, param, model);
    else if (param->svm_type == CSTREE_SVC || param->svm_type == CSFT_SVC)
        svm_train_cstree_csft(prob, param, model);
    else if (param->svm_type == CSAPFT_SVC)
        svm_train_csapft(prob, param, model);
    else if (param->svm_type == CSSECOC_SVC)
        svm_train_cssecoc(prob, param, model);
    else{
        fprintf(stderr,"svm_type not implemented.\n");
        exit(-1);
    }
    return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
    int i;
    int *fold_start = Malloc(int,nr_fold+1);
    int l = prob->l;
    int *perm = Malloc(int,l);
    int nr_class;

    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    if((param->svm_type == C_SVC ||
        param->svm_type == NU_SVC ||
        ISCSSVC(param->svm_type))
       && nr_fold < l)
    {
        int *start = NULL;
        int *label = NULL;
        int *count = NULL;
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

        // random shuffle and then data grouped by fold using the array perm
        int *fold_count = Malloc(int,nr_fold);
        int c;
        int *index = Malloc(int,l);
        for(i=0;i<l;i++)
            index[i]=perm[i];
        for (c=0; c<nr_class; c++) 
            for(i=0;i<count[c];i++)
            {
                int j = i+rand()%(count[c]-i);
                swap(index[start[c]+j],index[start[c]+i]);
            }
        for(i=0;i<nr_fold;i++)
        {
            fold_count[i] = 0;
            for (c=0; c<nr_class;c++)
                fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
        }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        for (c=0; c<nr_class;c++)
            for(i=0;i<nr_fold;i++)
            {
                int begin = start[c]+i*count[c]/nr_fold;
                int end = start[c]+(i+1)*count[c]/nr_fold;
                for(int j=begin;j<end;j++)
                {
                    perm[fold_start[i]] = index[j];
                    fold_start[i]++;
                }
            }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        free(start);      
        free(label);
        free(count);      
        free(index);
        free(fold_count);
    }
    else
    {
        for(i=0;i<l;i++) perm[i]=i;
        for(i=0;i<l;i++)
        {
            int j = i+rand()%(l-i);
            swap(perm[i],perm[j]);
        }
        for(i=0;i<=nr_fold;i++)
            fold_start[i]=i*l/nr_fold;
    }

    for(i=0;i<nr_fold;i++)
    {
        int begin = fold_start[i];
        int end = fold_start[i+1];
        int j,k;
        struct svm_problem subprob;
        int nr_class = prob->max_class;

        subprob.l = l-(end-begin);
        subprob.x = Malloc(struct svm_node*,subprob.l);
        subprob.y = Malloc(double,subprob.l);

        if (ISCSSVC(param->svm_type)){
            subprob.max_class = prob->max_class;
            subprob.cost = Malloc(double, subprob.l*nr_class);
        }
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            if (ISCSSVC(param->svm_type))
                memcpy(&(subprob.cost[k*nr_class]), &(prob->cost[perm[j]*nr_class]), sizeof(double)*nr_class);
            ++k;
        }
        for(j=end;j<l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            if (ISCSSVC(param->svm_type))
                memcpy(&(subprob.cost[k*nr_class]), &(prob->cost[perm[j]*nr_class]), sizeof(double)*nr_class);
            ++k;
        }
        struct svm_model *submodel = svm_train(&subprob,param);
        if(param->probability && 
           (param->svm_type == C_SVC || param->svm_type == NU_SVC))
        {
            double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
            for(j=begin;j<end;j++)
                target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
            free(prob_estimates);                 
        }
        else
            for(j=begin;j<end;j++){
                target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
            }
        svm_destroy_model(submodel);
        if (ISCSSVC(param->svm_type))
            free(subprob.cost);
        free(subprob.x);
        free(subprob.y);
    }           
    free(fold_start);
    free(perm);   
}


int svm_get_svm_type(const svm_model *model)
{
    return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
    return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
    if (model->label != NULL)
        for(int i=0;i<model->nr_class;i++)
            label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
    if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
        model->probA!=NULL)
        return model->probA[0];
    else
    {
        info("Model doesn't contain information for SVR probability inference\n");
        return 0;
    }
}

void svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;
        for(int i=0;i<model->l;i++)
            sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
        sum -= model->rho[0];
        *dec_values = sum;
    }
    else if (model->param.svm_type == CSOVO_SVC || model->param.svm_type == WAP_SVC)
        svm_predict_values_csovo_wap(model, x, dec_values);
    else if (model->param.svm_type == CSOVA_SVC)
        svm_predict_values_csova(model, x, dec_values);       
    else if (model->param.svm_type == CSPCR_ESVR || model->param.svm_type == CSOSR)
        svm_predict_values_cspcr_csosr(model, x, dec_values);         
    else if (model->param.svm_type == CSTREE_SVC || model->param.svm_type == CSFT_SVC)
        svm_predict_values_cstree_csft(model, x, dec_values);
    else if (model->param.svm_type == CSAPFT_SVC)
        svm_predict_values_csapft(model, x, dec_values);
    else if (model->param.svm_type == CSSECOC_SVC)
        svm_predict_values_cssecoc(model, x, dec_values);     
    else
    {
        int i;
        int nr_class = model->nr_class;
        int l = model->l;
	
        double *kvalue = Malloc(double,l);
#pragma omp parallel for private(i) 
        for(i=0;i<l;i++)
            kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        int p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];
                                
                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;
                p++;
            }

        free(kvalue);
        free(start);
    }
}

double svm_predict(const svm_model *model, const svm_node *x)
{
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        double res;
        svm_predict_values(model, x, &res);
                
        if(model->param.svm_type == ONE_CLASS)
            return (res>0)?1:-1;
        else
            return floor(res+0.5);
    }
    else if (model->param.svm_type == CSOVO_SVC || model->param.svm_type == WAP_SVC)
        return svm_predict_csovo_wap(model, x);
    else if (model->param.svm_type == CSOVA_SVC || model->param.svm_type == CSPCR_ESVR || model->param.svm_type == CSOSR)
        return svm_predict_csova_cspcr_csosr(model, x);
    else if (model->param.svm_type == CSTREE_SVC || model->param.svm_type == CSFT_SVC)
        return svm_predict_cstree_csft(model, x);
    else if (model->param.svm_type == CSAPFT_SVC)
	return svm_predict_csapft(model, x);
    else if (model->param.svm_type == CSSECOC_SVC)
        return svm_predict_cssecoc(model, x);    
    else //SVC
    {
        int i;
        int nr_class = model->nr_class;
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        svm_predict_values(model, x, dec_values);

        int *vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;
        int pos=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                if(dec_values[pos++] > 0)
                    ++vote[i];
                else
                    ++vote[j];
            }

        int vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;
	while(1){//a trick to randomly choose one when equal votes
	    i = rand() % nr_class;
	    if (vote[i] == vote[vote_max_idx]){
		vote_max_idx = i;
		break;
	    }
	}
        free(vote);
        free(dec_values);

        return model->label[vote_max_idx];
    }
}

double svm_predict_probability(
    const svm_model *model, const svm_node *x, double *prob_estimates)
{
    if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
        model->probA!=NULL && model->probB!=NULL)
    {
        int i;
        int nr_class = model->nr_class;
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        svm_predict_values(model, x, dec_values);

        double min_prob=1e-7;
        double **pairwise_prob=Malloc(double *,nr_class);
        for(i=0;i<nr_class;i++)
            pairwise_prob[i]=Malloc(double,nr_class);
        int k=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
                pairwise_prob[j][i]=1-pairwise_prob[i][j];
                k++;
            }
        multiclass_probability(nr_class,pairwise_prob,prob_estimates);

        int prob_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(prob_estimates[i] > prob_estimates[prob_max_idx])
                prob_max_idx = i;
        for(i=0;i<nr_class;i++)
            free(pairwise_prob[i]);
        free(dec_values);
        free(pairwise_prob);           
        return model->label[prob_max_idx];
    }
    else 
        return svm_predict(model, x);
}

const char *svm_type_table[] =
{
    "c_svc","nu_svc","one_class","epsilon_svr_round","nu_svr_round","csovo_svc","wap_svc","csova_svc","cspcr_esvr", "csosr", "cstree_svc", "csft_svc", "csapft_svc", "cssecoc_svc", NULL
};

const char *kernel_type_table[]=
{
    "linear","polynomial","rbf","sigmoid","stump","perc",
    "laplace","expo","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
    FILE *fp = fopen(model_file_name,"w");
    if(fp==NULL) return -1;

    const svm_parameter& param = model->param;

    fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
    fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

    if(param.kernel_type == POLY)
        fprintf(fp,"degree %d\n", param.degree);

    if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID 
       || param.kernel_type == LAPLACE || param.kernel_type == EXPO)
        fprintf(fp,"gamma %g\n", param.gamma);

    if(param.kernel_type == POLY || param.kernel_type == SIGMOID
       || param.kernel_type == STUMP || param.kernel_type == PERC)
        fprintf(fp,"coef0 %g\n", param.coef0);

    //INSERTED FOR COST-SENSITIVE VARIANTS
    if (model->param.svm_type == CSOVO_SVC || model->param.svm_type == WAP_SVC)
        return svm_save_model_csovo_wap(fp, model);
    else if (model->param.svm_type == CSOVA_SVC || model->param.svm_type == CSPCR_ESVR || model->param.svm_type == CSOSR)
        return svm_save_model_csova_cspcr_csosr(fp, model);
    else if (model->param.svm_type == CSTREE_SVC || model->param.svm_type == CSFT_SVC)
        return svm_save_model_cstree_csft(fp, model);
    else if (model->param.svm_type == CSAPFT_SVC)
        return svm_save_model_csapft(fp, model);
    else if (model->param.svm_type == CSSECOC_SVC)
        return svm_save_model_cssecoc(fp, model);

    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);
        
    {
        fprintf(fp, "rho");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }
        
    if(model->label)
    {
        fprintf(fp, "label");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }

    if(model->probA) // regression has probA only
    {
        fprintf(fp, "probA");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probA[i]);
        fprintf(fp, "\n");
    }
    if(model->probB)
    {
        fprintf(fp, "probB");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probB[i]);
        fprintf(fp, "\n");
    }

    if(model->nSV)
    {
        fprintf(fp, "nr_sv");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    const svm_node * const *SV = model->SV;

    for(int i=0;i<l;i++)
    {
        for(int j=0;j<nr_class-1;j++)
            fprintf(fp, "%.16g ",sv_coef[j][i]);

        const svm_node *p = SV[i];

        if(param.kernel_type == PRECOMPUTED)
            fprintf(fp,"0:%d ",(int)(p->value));
        else
            while(p->index != -1)
            {
                fprintf(fp,"%d:%.8g ",p->index,p->value);
                p++;
            }
        fprintf(fp, "\n");
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0;
}

svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"rb");
    if(fp==NULL) return NULL;
        
    // read parameters

    svm_model *model = Malloc(svm_model,1);
    svm_parameter& param = model->param;
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->label = NULL;
    model->nSV = NULL;

    char cmd[81];
    while(1)
    {
        fscanf(fp,"%80s",cmd);

        if(strcmp(cmd,"svm_type")==0)
        {
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;svm_type_table[i];i++)
            {
                if(strcmp(svm_type_table[i],cmd)==0)
                {
                    param.svm_type=i;
                    break;
                }
            }
                        
	    if (param.svm_type == CSOVO_SVC || param.svm_type == WAP_SVC){
                return svm_load_model_csovo_wap(fp, model);
            }
            else if(param.svm_type == CSOVA_SVC || param.svm_type == CSPCR_ESVR || param.svm_type == CSOSR){
                return svm_load_model_csova_cspcr_csosr(fp, model);
            }
            else if (param.svm_type == CSTREE_SVC || param.svm_type == CSFT_SVC){
                return svm_load_model_cstree_csft(fp, model);
            }
            else if (param.svm_type == CSAPFT_SVC){
                return svm_load_model_csapft(fp, model);
            }                         
            else if (param.svm_type == CSSECOC_SVC){
                return svm_load_model_cssecoc(fp, model);
            }            

            if(svm_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown svm type.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"kernel_type")==0)
        {               
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++)
            {
                if(strcmp(kernel_type_table[i],cmd)==0)
                {
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0)
            fscanf(fp,"%d",&model->nr_class);
        else if(strcmp(cmd,"total_sv")==0)
            fscanf(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0)
        {
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"probA")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probA = Malloc(double,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%lf",&model->probA[i]);
        }
        else if(strcmp(cmd,"probB")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probB = Malloc(double,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%lf",&model->probB[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0)
        {
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0)
        {
            while(1)
            {
                int c = getc(fp);
                if(c==EOF || c=='\n') break;      
            }
            break;
        }
        else
        {
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }

    // read sv_coef and SV

    int elements = 0;
    long pos = ftell(fp);

    while(1)
    {
        int c = fgetc(fp);
        switch(c){
        case '\n':
            // count the '-1' element
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
out:
    fseek(fp,pos,SEEK_SET);

    int m = model->nr_class - 1;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(svm_node*,l);
    svm_node *x_space=NULL;
    if(l>0) x_space = Malloc(svm_node,elements);

    int j=0;
    for(i=0;i<l;i++)
    {
        model->SV[i] = &x_space[j];
        for(int k=0;k<m;k++)
            fscanf(fp,"%lf",&model->sv_coef[k][i]);
        while(1)
        {
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
            ++j;
        }       
    out2:
        x_space[j++].index = -1;
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

    model->free_sv = 1;   // XXX
    return model;
}

void svm_destroy_model(svm_model* model)
{
    if(model->free_sv && model->l > 0)
        free((void *)(model->SV[0]));
    for(int i=0;i<model->nr_class-1;i++)
        free(model->sv_coef[i]);
    free(model->SV);
    free(model->sv_coef);
    free(model->rho);
    free(model->label);
    free(model->probA);
    free(model->probB);
    free(model->nSV);
    free(model);
}

void svm_destroy_param(svm_parameter* param)
{
    free(param->weight_label);
    free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
    // svm_type

    int svm_type = param->svm_type;
    if(svm_type != C_SVC && 
       svm_type != NU_SVC &&
       svm_type != ONE_CLASS &&
       svm_type != EPSILON_SVR &&
       svm_type != NU_SVR &&
       !ISCSSVC(svm_type))
        return "unknown svm type";
        
    // kernel_type, degree
        
    int kernel_type = param->kernel_type;
    if(kernel_type != LINEAR &&
       kernel_type != POLY &&
       kernel_type != RBF &&
       kernel_type != SIGMOID &&
       kernel_type != STUMP &&
       kernel_type != PERC &&
       kernel_type != LAPLACE &&
       kernel_type != EXPO &&
       kernel_type != PRECOMPUTED)
        return "unknown kernel type";

    if(param->degree < 0)
        return "degree of polynomial kernel < 0";

    // cache_size,eps,C,nu,p,shrinking

    if(param->cache_size <= 0)
        return "cache_size <= 0";

    if(param->eps <= 0)
        return "eps <= 0";

    if(svm_type == C_SVC ||
       svm_type == EPSILON_SVR ||
       svm_type == NU_SVR)
        if(param->C <= 0)
            return "C <= 0";

    if(svm_type == NU_SVC ||
       svm_type == ONE_CLASS ||
       svm_type == NU_SVR)
        if(param->nu <= 0 || param->nu > 1)
            return "nu <= 0 or nu > 1";

    if(svm_type == EPSILON_SVR)
        if(param->p < 0)
            return "p < 0";

    if(param->shrinking != 0 &&
       param->shrinking != 1)
        return "shrinking != 0 and shrinking != 1";

    if(param->probability != 0 &&
       param->probability != 1)
        return "probability != 0 and probability != 1";

    if(param->probability == 1 &&
       svm_type == ONE_CLASS)
        return "one-class SVM probability output not supported yet";


    // check whether nu-svc is feasible
        
    if(svm_type == NU_SVC)
    {
        int l = prob->l;
        int max_nr_class = 16;
        int nr_class = 0;
        int *label = Malloc(int,max_nr_class);
        int *count = Malloc(int,max_nr_class);

        int i;
        for(i=0;i<l;i++)
        {
            int this_label = (int)prob->y[i];
            int j;
            for(j=0;j<nr_class;j++)
                if(this_label == label[j])
                {
                    ++count[j];
                    break;
                }
            if(j == nr_class)
            {
                if(nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    label = (int *)realloc(label,max_nr_class*sizeof(int));
                    count = (int *)realloc(count,max_nr_class*sizeof(int));
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }
        
        for(i=0;i<nr_class;i++)
        {
            int n1 = count[i];
            for(int j=i+1;j<nr_class;j++)
            {
                int n2 = count[j];
                if(param->nu*(n1+n2)/2 > min(n1,n2))
                {
                    free(label);
                    free(count);
                    return "specified nu is infeasible";
                }
            }
        }
        free(label);
        free(count);
    }

    return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
    return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
            model->probA!=NULL && model->probB!=NULL) ||
        ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
         model->probA!=NULL);
}

//====BEGIN OF CSOVO_WAP====
inline void compute_v_matrix_wap(double* cost, double* v, int& l, int& nr_class){
  
    double *costnow = Malloc(double, nr_class);
    double *vnow = Malloc(double, nr_class);
    for(int n=0;n<l;n++){             
        int shift = n * nr_class;
    
        memcpy(costnow, cost + shift, sizeof(double) * nr_class);
    
        std::sort(costnow, costnow+nr_class);
    
        for(int i=nr_class-1;i>=1;i--)
            vnow[i] = costnow[i] - costnow[i-1];
    
        vnow[0] = 0.0;
        for(int i=1;i<nr_class;i++){
            vnow[i] /= i;
            vnow[i] += vnow[i-1];
        }
    
        for(int i=0;i<nr_class;i++){
            for(int k=0;k<nr_class;k++){
                if (cost[shift + i] == costnow[k]){
                    v[shift + i] = vnow[k];
                    break;
                }
            }
        }
    }
    free(vnow);
    free(costnow);
}

void svm_train_csovo_wap(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    // CSOVO_SVC or WAP_SVC classification
    int l = prob->l;
    int nr_class = prob->max_class;
  
    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    // train k*(k-1)/2 models     
    bool *nonzero = Malloc(bool, l*nr_class); //recording whether an example is a SV for class k
    memset(nonzero, false, sizeof(bool)*l*nr_class);
    decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

    double *C = Malloc(double, l); //the C for subproblems
    int *idx = Malloc(int, l);     //the original index of examples
  
    double* v;
    if (param->svm_type == WAP_SVC){
        v = Malloc(double, l*nr_class);             
        compute_v_matrix_wap(prob->cost, v, l, nr_class);
    }
    else{
        v = prob->cost;
    }

    {
	svm_problem sub_prob;
	sub_prob.x = Malloc(svm_node *, l);
	sub_prob.y = Malloc(double, l);

        int p = 0;
        for(int i=0;i<nr_class;i++){
            for(int j=i+1;j<nr_class;j++){
                int m = 0;                              
                for(int n=0;n<l;n++){
                    C[m] = fabs(v[n*nr_class+i] - v[n*nr_class+j]) * param->C;
          
                    idx[m] = n;
                    if (C[m] > 0)
                        m++;
                }
                sub_prob.l = m;
        
                for(m=0;m<sub_prob.l;m++){
                    int n = idx[m];
                    sub_prob.x[m] = prob->x[n];
                    sub_prob.y[m] = (prob->cost[n*nr_class+i] < prob->cost[n*nr_class+j] ? +1 : -1);
                }
        
                f[p] = svm_train_one(&sub_prob,param,C);
        
                f[p].idx = Malloc(int, sub_prob.l);
                memcpy(f[p].idx, idx, sizeof(int)*sub_prob.l);
        
                f[p].len = sub_prob.l;
        
                for(m=0;m<sub_prob.l;m++){
                    int n = idx[m];
                    if (sub_prob.y[m] > 0)
                        nonzero[n*nr_class+i] |= 
                            (fabs(f[p].alpha[m]) > 0);
                    else
                        nonzero[n*nr_class+j] |= 
                            (fabs(f[p].alpha[m]) > 0);
                }
                ++p;
            }
        }

	free(sub_prob.y);
	free(sub_prob.x);
	
    }


    if (param->svm_type == WAP_SVC){
        free(v);
    }
  
    free(idx);
    free(C);
  
    // build output
    model->nr_class = nr_class;
  
    model->label = Malloc(int,nr_class);
    for(int k=0;k<nr_class;k++)
        model->label[k] = k+1;
  
    model->rho = Malloc(double,nr_class*(nr_class-1)/2);
    for(int p=0;p<nr_class*(nr_class-1)/2;p++)
        model->rho[p] = f[p].rho;
  
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class);
    for(int k=0;k<nr_class;k++){
        int nSV = 0;                        
        for(int n=0;n<l;n++)
            if(nonzero[n*nr_class+k])
                ++nSV;
    
        model->nSV[k] = nSV;
        total_sv += nSV;
    }
                
    info("Total nSV = %d\n",total_sv);
                
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    idx = Malloc(int, l*nr_class);
    int *nz_start = Malloc(int,nr_class);
    {
        int p = 0;
        for(int k=0;k<nr_class;k++){ //class first to group SV for each class together
            nz_start[k] = p;
            for(int n=0;n<l;n++){
                if(nonzero[n*nr_class+k]){
                    model->SV[p] = prob->x[n];
                    idx[n*nr_class+k] = p;
                    p++;
                }
                else
                    idx[n*nr_class+k] = -1;
            }
        }
    }

    model->sv_coef = Malloc(double *,nr_class-1);
    for(int k=0;k<nr_class-1;k++){
        model->sv_coef[k] = Malloc(double,total_sv);
        memset(model->sv_coef[k], 0, sizeof(double)*total_sv);
    }

    {             
        int p = 0;
        for(int i=0;i<nr_class;i++){
            for(int j=i+1;j<nr_class;j++){
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]
                for(int m=0;m<f[p].len;m++){
                    int n = f[p].idx[m];
                    if (nonzero[n*nr_class+i]){
                        int ref = idx[n*nr_class+i];
                        model->sv_coef[j-1][ref] = f[p].alpha[m];
                        f[p].alpha[m] = 0; //to avoid multiple inclusion
                    }
                }
      
                for(int m=0;m<f[p].len;m++){
                    int n = f[p].idx[m];                      
                    if (nonzero[n*nr_class+j]){
                        int ref = idx[n*nr_class+j];
                        model->sv_coef[i][ref] = f[p].alpha[m];
                        f[p].alpha[m] = 0; //to avoid multiple inclusion
                    }
                }
                ++p;
            }
        }
    }

    free(nz_start);
    free(idx);
  
    for(int p=0;p<nr_class*(nr_class-1)/2;p++){
        free(f[p].idx);
        free(f[p].alpha);
    }
    free(f);

    free(nonzero);
}

void svm_predict_values_csovo_wap(const svm_model *model, const svm_node *x, double* dec_values){
    //copied from the original OVO-SVM
    int i;
    int nr_class = model->nr_class;
    int l = model->l;
  
    double *kvalue = Malloc(double,l);
#pragma omp parallel for private(i) 
    for(i=0;i<l;i++)
        kvalue[i] = Kernel::k_function(x, model->SV[i], model->param);
  
    int *start = Malloc(int,nr_class);
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+model->nSV[i-1];
  
    int p=0;
    for(i=0;i<nr_class;i++)
        for(int j=i+1;j<nr_class;j++){
            double sum = 0;
            int si = start[i];
            int sj = start[j];
            int ci = model->nSV[i];
            int cj = model->nSV[j];
      
            int k;
            double *coef1 = model->sv_coef[j-1];
            double *coef2 = model->sv_coef[i];
            for(k=0;k<ci;k++)
                sum += coef1[si+k] * kvalue[si+k];
            for(k=0;k<cj;k++)
                sum += coef2[sj+k] * kvalue[sj+k];
            sum -= model->rho[p];
            dec_values[p] = sum;
            p++;
        }
  
    free(kvalue);
    free(start);  
}

double svm_predict_csovo_wap(const svm_model *model, const svm_node *x){
    //copied from the original OVO-SVM
    int i;
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
    svm_predict_values(model, x, dec_values);

    int *vote = Malloc(int,nr_class);
    for(i=0;i<nr_class;i++)
        vote[i] = 0;
    int pos=0;
    for(i=0;i<nr_class;i++)
        for(int j=i+1;j<nr_class;j++){
            if(dec_values[pos++] > 0)
                ++vote[i];
            else
                ++vote[j];
        }
  
    int vote_max_idx = 0;
    for(i=1;i<nr_class;i++)
        if(vote[i] > vote[vote_max_idx])
            vote_max_idx = i;
    while(1){//a trick to randomly choose one when equal votes
	i = rand() % nr_class;
	if (vote[i] == vote[vote_max_idx]){
	    vote_max_idx = i;
	    break;
	}
    }
    
    free(vote);
    free(dec_values);

    return model->label[vote_max_idx];
}

svm_model *svm_load_model_csovo_wap(FILE* fp, svm_model* model){
    //modified from the original OVO-SVM
    if(fp==NULL) return NULL;
  
    // read parameters
    svm_parameter& param = model->param;
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->label = NULL;
    model->nSV = NULL;
  
    char cmd[81];
    while(1){
        fscanf(fp,"%80s",cmd);
    
        if(strcmp(cmd,"kernel_type")==0){
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++){
                if(strcmp(kernel_type_table[i],cmd)==0){
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL){
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0)
            fscanf(fp,"%d",&model->nr_class);
        else if(strcmp(cmd,"total_sv")==0)
            fscanf(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0){
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0){
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0){
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0;i<n;i++)
                fscanf(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0){
            while(1){
                int c = getc(fp);
                if(c==EOF || c=='\n') break;    
            }
            break;
        }
        else{
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }

    // read sv_coef and SV
    
    int elements = 0;
    long pos = ftell(fp);
    
    while(1){
        int c = fgetc(fp);
        switch(c){
        case '\n':
            // count the '-1' element
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
out:
    fseek(fp,pos,SEEK_SET);
    
    int m = model->nr_class - 1;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(svm_node*,l);
    svm_node *x_space=NULL;
    if(l>0) 
        x_space = Malloc(svm_node,elements);
    
    int j=0;
    for(i=0;i<l;i++){
        model->SV[i] = &x_space[j];
        for(int k=0;k<m;k++)
            fscanf(fp,"%lf",&model->sv_coef[k][i]);
        while(1){
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
            ++j;
        }   
    out2:
        x_space[j++].index = -1;
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;
    
    model->free_sv = 1;   // XXX
    return model;
}

int svm_save_model_csovo_wap(FILE* fp, const svm_model *model){
    if(fp==NULL) return -1;
  
    const svm_parameter& param = model->param;
  
    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);
  
    {
        fprintf(fp, "rho");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }
  
    if(model->label){
        fprintf(fp, "label");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }

    if(model->nSV){
        fprintf(fp, "nr_sv");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    const svm_node * const *SV = model->SV;
  
    for(int i=0;i<l;i++){
        for(int j=0;j<nr_class-1;j++)
            fprintf(fp, "%.16g ",sv_coef[j][i]);
    
        const svm_node *p = SV[i];
    
        if(param.kernel_type == PRECOMPUTED)
            fprintf(fp,"0:%d ",(int)(p->value));
        else
            while(p->index != -1)
            {
                fprintf(fp,"%d:%.8g ",p->index,p->value);
                p++;
            }
        fprintf(fp, "\n");
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0;
}
//====END OF CSOVO_WAP====

//====BEGIN OF CSOVA====
inline svm_model *svm_load_model_simple(FILE* fp, svm_model* model, int n_label, int n_svm){
    if(fp==NULL) return NULL;
    
    // read parameters
    svm_parameter& param = model->param;
    
    char cmd[81];
    while(1){
        fscanf(fp,"%80s",cmd);
    
        if(strcmp(cmd,"kernel_type")==0){
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++){
                if(strcmp(kernel_type_table[i],cmd)==0){
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL){
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"total_sv")==0)
            fscanf(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0){
            model->rho = Malloc(double,n_svm);
            for(int i=0;i<n_svm;i++)
                fscanf(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0){
            model->label = Malloc(int,n_label);
            for(int i=0;i<n_label;i++)
                fscanf(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0){
            model->nSV = Malloc(int,n_svm);
            for(int i=0;i<n_svm;i++)
                fscanf(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0){
            while(1){
                int c = getc(fp);
                if(c==EOF || c=='\n') break;    
            }
            break;
        }
        else{
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }

    // read sv_coef and SV
  
    int elements = 0;
    long pos = ftell(fp);
  
    while(1){
        int c = fgetc(fp);
        switch(c){
        case '\n':
            // count the '-1' element
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
out:
    fseek(fp,pos,SEEK_SET);
  
    int m = n_svm;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,model->nSV[i]);
    model->SV = Malloc(svm_node*,l);
    svm_node *x_space=NULL;
    if(l>0) x_space = Malloc(svm_node,elements);
  
    int j=0;
    int jj=0;
    for(i=0;i<m;i++){
        for(int k=0;k<model->nSV[i];k++){
            model->SV[jj++] = &x_space[j];
            fscanf(fp,"%lf",&model->sv_coef[i][k]);
            while(1){
                int c;
                do {
                    c = getc(fp);
                    if(c=='\n') goto out2;
                } while(isspace(c));
                ungetc(c,fp);
                fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
                ++j;
            } 
        out2:
            x_space[j++].index = -1;
        }
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;
  
    model->free_sv = 1;   // XXX
    
    return model;
}

inline int svm_save_model_simple(FILE* fp, const svm_model *model, int n_label, int n_svm){
    if(fp==NULL) return -1;
    const svm_parameter& param = model->param;
  
    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);

    {
        fprintf(fp, "rho");
        for(int i=0;i<n_svm;i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }
  
    if(model->label){
        fprintf(fp, "label");
        for(int i=0;i<n_label;i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }
  
    if(model->nSV){
        fprintf(fp, "nr_sv");
        for(int i=0;i<n_svm;i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }
  
    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    const svm_node * const *SV = model->SV;
  
    int q=0;
    for(int i=0;i<n_svm;i++){
        for(int m=0;m<model->nSV[i];m++){
            fprintf(fp, "%.16g ",sv_coef[i][m]);
      
            const svm_node *p = SV[q++];
      
            if(param.kernel_type == PRECOMPUTED)
                fprintf(fp,"0:%d ",(int)(p->value));
            else
                while(p->index != -1){
                    fprintf(fp,"%d:%.8g ",p->index,p->value);
                    p++;
                }
            fprintf(fp, "\n");
        }
    }
    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0;
}

void svm_train_csova(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    int l = prob->l;
    int nr_class = prob->max_class;
  
    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    // train k models     
    double *C = Malloc(double, l);
    int *idx = Malloc(int, l);
  
    decision_function *f = Malloc(decision_function,nr_class);
  
    double *maxcost = Malloc(double, l);
  
    for(int n=0;n<l;n++){
        double* pcost = &(prob->cost[n*nr_class]);
        maxcost[n] = (*pcost);
        for(int i=1;i<nr_class;i++)
            maxcost[n] = max(maxcost[n], pcost[i]);
    }
  
    svm_problem sub_prob;
    sub_prob.x = Malloc(svm_node *,l);
    sub_prob.y = Malloc(double, l);
  
    for(int i=0;i<nr_class;i++){
        int m = 0;                          
        for(int n=0;n<l;n++){
      
            if (prob->y[n] == i+1){
                sub_prob.y[m] = +1;
                C[m] = (maxcost[n] - prob->cost[n*nr_class+i]) / maxcost[n];
            }
            else{
                sub_prob.y[m] = -1;
                C[m] = prob->cost[n*nr_class+i] / maxcost[n];
            }
      
            if (fabs(C[m]) > 0.0){
                sub_prob.x[m] = prob->x[n];
                C[m] *= param->C;
                C[m] *= maxcost[n];
                idx[m] = n;
                m++;
            }
        }
        sub_prob.l = m;
        f[i] = svm_train_one(&sub_prob,param,C);
    
        f[i].len = m;
        f[i].idx = Malloc(int, f[i].len);
        memcpy(f[i].idx, idx, sizeof(int)*f[i].len);
    
    }
    free(sub_prob.y);
    free(sub_prob.x);

    free(maxcost);
  
    free(idx);
    free(C);
  
    // build output
    model->nr_class = nr_class;
  
    model->label = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++)
        model->label[i] = i+1;
  
    model->rho = Malloc(double,nr_class);
    for(int i=0;i<nr_class;i++)
        model->rho[i] = f[i].rho;
  
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++){
        int nSV = 0;                        
        for(int n=0;n<f[i].len;n++)
            if(fabs(f[i].alpha[n]) > 0)
                ++nSV;
    
        model->nSV[i] = nSV;
        total_sv += nSV;
    }
  
    info("Total nSV = %d\n",total_sv);
  
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,nr_class);
  
    int p = 0;
    for(int i=0;i<nr_class;i++){
        model->sv_coef[i] = Malloc(double,model->nSV[i]);
        int m = 0;
        for(int n=0;n<f[i].len;n++){
            if(fabs(f[i].alpha[n]) > 0){
                model->SV[p++] = prob->x[f[i].idx[n]];
                model->sv_coef[i][m++] = f[i].alpha[n];
            }
        }
    }
  
    for(int i=0;i<nr_class;i++){
        free(f[i].idx);
        free(f[i].alpha);
    }
    free(f);
}

void svm_predict_values_csova(const svm_model *model, const svm_node *x, double* dec_values){  
    int i;
    int nr_class = model->nr_class;
  
    int start = 0;
    for(i=0;i<nr_class;i++){
        double res = -model->rho[i];
        for(int j=0;j<model->nSV[i];j++){
            res += model->sv_coef[i][j] * Kernel::k_function(x,model->SV[start+j],model->param);
        }
        dec_values[i] = res;
        start += model->nSV[i];
    }
}

double svm_predict_csova_cspcr_csosr(const svm_model *model, const svm_node *x){
    int i;
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, nr_class);
    svm_predict_values(model, x, dec_values);
  
    int max_idx = 0;
    for(i=0;i<nr_class;i++){
        if (dec_values[i] > dec_values[max_idx])
            max_idx = i;
    }
    while(1){//a trick to randomly choose one when equal values
	i = rand() % nr_class;
	if (dec_values[i] == dec_values[max_idx]){
	    max_idx = i;
	    break;
	}
    }

    free(dec_values);
    return model->label[max_idx];
}

svm_model *svm_load_model_csova_cspcr_csosr(FILE* fp, svm_model* model){
    if(fp==NULL) return NULL;
  
    // read parameters
    svm_parameter& param = model->param;
  
    char cmd[81];
    while(1){
        fscanf(fp,"%80s",cmd);
    
        if(strcmp(cmd,"kernel_type")==0){
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++){
                if(strcmp(kernel_type_table[i],cmd)==0){
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL){
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0){
            fscanf(fp,"%d",&model->nr_class);
	    return svm_load_model_simple(fp, model, model->nr_class, model->nr_class);
	}
        else{
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }
    return NULL;
}

int svm_save_model_csova_cspcr_csosr(FILE* fp, const svm_model *model){
    if(fp==NULL) return -1;
    int nr_class = model->nr_class;
  
    return svm_save_model_simple(fp, model, nr_class, nr_class);
}
//====END OF CSOVA====

//====BEGIN OF CSPCR====
void svm_train_cspcr(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    int l = prob->l;
    int nr_class = prob->max_class;

    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }

    // train nr_class models        
    decision_function *f = Malloc(decision_function,nr_class);
    svm_parameter *sub_param = Malloc(svm_parameter, 1);
    memcpy(sub_param, param, sizeof(svm_parameter));
    sub_param->svm_type = EPSILON_SVR;

    svm_problem sub_prob;
    sub_prob.x = Malloc(svm_node *,l);
    sub_prob.y = Malloc(double, l);

    double maxcost = 0.0;
  
    for(int n=0;n<l;n++){
        for(int i=1;i<nr_class;i++)
            maxcost = max(maxcost, fabs(prob->cost[n*nr_class+i]));
    }
    if (maxcost <= 0.0)
        maxcost = 1.0;

    for(int i=0;i<nr_class;i++){
        for(int n=0;n<l;n++){
            sub_prob.y[n] = prob->cost[n*nr_class+i] / maxcost;
            sub_prob.x[n] = prob->x[n];
        }
        sub_prob.l = l;
        f[i] = svm_train_one(&sub_prob,sub_param,param->C,param->C);    
    }

    free(sub_prob.x);
    free(sub_prob.y);
    free(sub_param);

    // build output
    model->nr_class = nr_class;
  
    model->label = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++)
        model->label[i] = i+1;
                
    model->rho = Malloc(double,nr_class);
    for(int i=0;i<nr_class;i++)
        model->rho[i] = f[i].rho;
                
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++){
        int nSV = 0;                        
        for(int n=0;n<l;n++)
            if(fabs(f[i].alpha[n]) > 0)
                ++nSV;
    
        model->nSV[i] = nSV;
        total_sv += nSV;
    }
  
    info("Total nSV = %d\n",total_sv);
  
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,nr_class);
  
    {int p = 0;
        for(int i=0;i<nr_class;i++){
            model->sv_coef[i] = Malloc(double,model->nSV[i]);
            int m = 0;
            for(int n=0;n<l;n++){
                if(fabs(f[i].alpha[n]) > 0){
                    model->SV[p++] = prob->x[n];
                    model->sv_coef[i][m++] = f[i].alpha[n];
                }
            }
        }
    }

    for(int i=0;i<nr_class;i++){
        free(f[i].alpha);
    }
    free(f);
}

void svm_predict_values_cspcr_csosr(const svm_model *model, const svm_node *x, double* dec_values){  
    int i;
    int nr_class = model->nr_class;
  
    int start = 0;
    for(i=0;i<nr_class;i++){
        double res = -model->rho[i];
        for(int j=0;j<model->nSV[i];j++){
            res += model->sv_coef[i][j] * Kernel::k_function(x,model->SV[start+j],model->param);
        }
        dec_values[i] = -res;
        start += model->nSV[i];
    }
}
//====END OF CSPCR====

//====BEGIN OF CSOSR====
void svm_train_csosr(const svm_problem *prob, const svm_parameter *param, svm_model *model)
{
    int l = prob->l;
    int nr_class = prob->max_class;

    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    // train k models     
    double *RHS = Malloc(double, l);
    double* C = Malloc(double, l);
    int *idx = Malloc(int, l);
  
    decision_function *f = Malloc(decision_function,nr_class);
  
    double *maxcost = Malloc(double, l);
    double *mincost = Malloc(double, l);
  
    for(int n=0;n<l;n++){
        double* pcost = &(prob->cost[n*nr_class]);
        maxcost[n] = (*pcost);
        mincost[n] = (*pcost);
        for(int i=1;i<nr_class;i++){
            maxcost[n] = max(maxcost[n], pcost[i]);
            mincost[n] = min(mincost[n], pcost[i]);
        }
    }

    svm_problem sub_prob;
    sub_prob.x = Malloc(svm_node *,l);
    sub_prob.y = Malloc(double, l);
  
    for(int i=0;i<nr_class;i++){
        for(int n=0;n<l;n++){
            if (prob->cost[n*nr_class+i] == mincost[n]){
                sub_prob.y[n] = -1;
                RHS[n] = - prob->cost[n*nr_class+i] / maxcost[n];
            }
            else{
                sub_prob.y[n] = +1;
                RHS[n] = prob->cost[n*nr_class+i] / maxcost[n];
            }
      
            sub_prob.x[n] = prob->x[n];
            C[n] = maxcost[n] * param->C;
            idx[n] = n;
        }
        sub_prob.l = l;
        f[i] = svm_train_one(&sub_prob,param,RHS,C);
    
        f[i].len = sub_prob.l;
        f[i].idx = Malloc(int, f[i].len);
        memcpy(f[i].idx, idx, sizeof(int)*f[i].len);
    
    }
    free(mincost);
    free(maxcost);

    free(sub_prob.x);
    free(sub_prob.y);

    free(RHS);
    free(C);
    free(idx);
  
    // build output
    model->nr_class = nr_class;
  
    model->label = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++)
        model->label[i] = i+1;
                
    model->rho = Malloc(double,nr_class);
    for(int i=0;i<nr_class;i++)
        model->rho[i] = f[i].rho;
  
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++){
        int nSV = 0;                        
        for(int n=0;n<f[i].len;n++)
            if(fabs(f[i].alpha[n]) > 0)
                ++nSV;
    
        model->nSV[i] = nSV;
        total_sv += nSV;
    }
  
    info("Total nSV = %d\n",total_sv);
  
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,nr_class);
  
    {
        int p = 0;
        for(int i=0;i<nr_class;i++){
            model->sv_coef[i] = Malloc(double,model->nSV[i]);
            int m = 0;
            for(int n=0;n<f[i].len;n++){
                if(fabs(f[i].alpha[n]) > 0){
                    model->SV[p++] = prob->x[f[i].idx[n]];
                    model->sv_coef[i][m++] = f[i].alpha[n];
                }
            }
        }
    }

    for(int i=0;i<nr_class;i++){
        free(f[i].idx);
        free(f[i].alpha);
    }
    free(f);
}
//====END OF CSOSR====

//====BEGIN OF CSTREE/CSFT====
void svm_train_cstree_csft(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    int l = prob->l;
    int nr_class = prob->max_class;
  
    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    int *perm = Malloc(int,nr_class);
  
    // random shuffle
    for(int i=0;i<nr_class;i++) 
        perm[i]=i;
    for(int i=0;i<nr_class;i++){
        int j = i+rand()%(nr_class-i);
        swap(perm[i],perm[j]);
    }
  
    int *tree = Malloc(int, (nr_class-1)*2);
  
    for(int i=0;i<nr_class;i++)
        tree[i] = perm[i]+1;
  
    int begin = 0;
    int end = nr_class;
    do{     
        int num = end - begin;
        if (num % 2 == 1){
            tree[end+num/2-1] = tree[end-1]; //shift the singleton to the final position
            end--; num--;
      
            for(int i = 0; i<num/2; i++)
                tree[end+i] = -(begin + i*2); 
      
            begin = end;
            end = end + num/2 + 1; //add the singleton back.
        }
        else{
            for(int i = 0; i<num/2; i++)
                tree[end+i] = -(begin + i*2);
            begin = end;
            end = end + num/2;
        }
    }while(end-begin > 2);
  
          
    // train k-1 models   
    double *C = Malloc(double, l);
    int *idx = Malloc(int, l);
    decision_function *f = Malloc(decision_function, nr_class-1);
  
    double *ctmp = Malloc(double, l * nr_class);
    memcpy(ctmp, prob->cost, sizeof(double)* l * nr_class);
  
    svm_problem sub_prob;
          
    sub_prob.x = Malloc(svm_node *,l);
    sub_prob.y = Malloc(double, l);
    {
        int p = 0;
        int match=1;
    
        while(match < nr_class){
            for(int right = match; right < nr_class; right += match*2){
                int left = right - match;
        
                int m = 0;
        
                for(int n=0;n<l;n++){
                    double* pleft = &(ctmp[n * nr_class + perm[left]]);
                    double* pright = &(ctmp[n * nr_class + perm[right]]);
          
                    C[m] =  (*pleft) - (*pright);
          
                    if (fabs(C[m]) > 0.0){
                        sub_prob.x[m] = prob->x[n];
                        sub_prob.y[m] = (C[m] >= 0 ? +1 : -1);
                        C[m] *= param->C;
                        C[m] = fabs(C[m]);                    
                        idx[m] = n;
                        m++;
                    }
                }
                sub_prob.l = m;
        
                f[p] = svm_train_one(&sub_prob,param,C);
        
                f[p].len = m;
                f[p].idx = Malloc(int, f[p].len);
                memcpy(f[p].idx, idx, sizeof(int)*f[p].len);
        
                if (param->svm_type == CSTREE_SVC){
                    //swap by ideal cost
                    for(int n=0;n<l;n++){
                        double* pleft = &(ctmp[n * nr_class + perm[left]]);
                        double* pright = &(ctmp[n * nr_class + perm[right]]);
                        if ((*pright) < (*pleft))
                            (*pleft) = (*pright);
                    }
                }
                else{//CSFT_SVC
                    //swap by prediction cost
                    for(int n=0;n<l;n++){
                        double res = -f[p].rho;
                        for(int j=0;j<f[p].len;j++){
                            if (f[p].alpha[j] != 0.0)
                                res += f[p].alpha[j] * Kernel::k_function(prob->x[n], prob->x[f[p].idx[j]], *param);
                        }
            
                        if (res > 0 || (fabs(res) == 0 && (rand() & 1))){
                            double* pleft = &(ctmp[n * nr_class + perm[left]]);
                            double* pright = &(ctmp[n * nr_class + perm[right]]);
                            (*pleft) = (*pright);
                        }
                    }
                }
                p++;                                                    
            }
            match *= 2;
        }
    }

    free(sub_prob.x);
    free(sub_prob.y);
    free(C);
    free(idx);
    free(ctmp);
    free(perm);

    // build output
    model->nr_class = nr_class;
  
    model->label = tree;
  
    model->rho = Malloc(double,nr_class-1);
    for(int i=0;i<nr_class-1;i++)
        model->rho[i] = f[i].rho;
  
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class-1);
    for(int i=0;i<nr_class-1;i++){
        int nSV = 0;                        
        for(int n=0;n<f[i].len;n++)
            if(fabs(f[i].alpha[n]) > 0)
                ++nSV;
    
        model->nSV[i] = nSV;
        total_sv += nSV;
    }
  
    info("Total nSV = %d\n",total_sv);
  
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,nr_class-1);
  
    for(int i=0, p=0;i<nr_class-1;i++){
        model->sv_coef[i] = Malloc(double,model->nSV[i]);
        int m = 0;
        for(int n=0;n<f[i].len;n++){
            if(fabs(f[i].alpha[n]) > 0){
                model->SV[p++] = prob->x[f[i].idx[n]];
                model->sv_coef[i][m++] = f[i].alpha[n];
            }
        }
    }
    for(int i=0;i<nr_class-1;i++){
        free(f[i].idx);
        free(f[i].alpha);
    }
    free(f);
}

void svm_predict_values_cstree_csft(const svm_model *model, const svm_node *x, double* dec_values){
    int nr_class = model->nr_class;
  
    int start = 0;
    for(int i=0;i<nr_class-1;i++){
        double res = -model->rho[i];
        for(int j=0;j<model->nSV[i];j++){
            res += model->sv_coef[i][j] * Kernel::k_function(x,model->SV[start+j],model->param);
        }
        dec_values[i] = res;
        start += model->nSV[i];
    }
}

double svm_predict_cstree_csft(const svm_model *model, const svm_node *x){
    int nr_class = model->nr_class;
  
    double *dec_values = Malloc(double, nr_class-1);
    svm_predict_values(model, x, dec_values);
  
    int current = - (2 * (nr_class - 1) - 2);
  
    while(current <= 0){
        double res = dec_values[-current / 2];
        current = (res > 0 || (fabs(res) == 0 && (rand() & 1))
                   ? model->label[-current+1]
                   : model->label[-current]);
    }
    free(dec_values);
    return current;
}

svm_model *svm_load_model_cstree_csft(FILE* fp, svm_model* model){
    if(fp==NULL) return NULL;
  
    // read parameters
    svm_parameter& param = model->param;
  
    char cmd[81];
    while(1){
        fscanf(fp,"%80s",cmd);
    
        if(strcmp(cmd,"kernel_type")==0){
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++){
                if(strcmp(kernel_type_table[i],cmd)==0){
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL){
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0){
            fscanf(fp,"%d",&model->nr_class);
	    int nr_class = model->nr_class;	    
	    return svm_load_model_simple(fp, model, (nr_class-1)*2, nr_class-1);
	}
        else{
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }
    return NULL;
}

int svm_save_model_cstree_csft(FILE* fp, const svm_model *model){
    if(fp==NULL) return -1;
    int nr_class = model->nr_class;
  
    return svm_save_model_simple(fp, model, (nr_class-1)*2, nr_class-1);
}
//====END OF CSTREE/CSFT====

//====BEGIN OF CSAPTREE====
void svm_train_csapft(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    int l = prob->l;
    int nr_class = prob->max_class;
  
    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    int *label = Malloc(int,nr_class);
  
    // random shuffle
    for(int i=0;i<nr_class;i++) 
        label[i]=i;
    for(int i=0;i<nr_class;i++){
        int j = i+rand()%(nr_class-i);
        swap(label[i],label[j]);
    }
  
    //start
    int *games = Malloc(int, l*nr_class);
    for(int p=0, n=0;n<l;n++){
        for(int i=0;i<nr_class;i++){
            games[p] = label[i];
            p++;
        }
    }

    // train k*(k-1)/2 models     
    double *C = Malloc(double, l);
    int *idx = Malloc(int, l);

    bool *nonzero = Malloc(bool, l*nr_class); //recording whether an example is a SV for class k
    memset(nonzero, false, sizeof(bool)*l*nr_class);  
    decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);
    memset(f, 0, sizeof(decision_function)*nr_class*(nr_class-1)/2);
    svm_problem sub_prob;
    sub_prob.x = Malloc(svm_node*, l);
    sub_prob.y = Malloc(double, l);
  
    int level = 1;
  
    while(level <= nr_class){
        for(int i=0;i+level<nr_class;i+=(level*2)){
            for(int n=0;n<l;n++){       
                int one = games[n*nr_class+i];
                int two = games[n*nr_class+i+level];
                if (one > two)
                    swap(one, two);
                int pos = (nr_class - 1 + nr_class - one + 1 - 1) * (one) / 2;
                pos += (two - one - 1);
        
                if (!f[pos].alpha){
                    //Training one versus two
                    int m=0;
                    for(int nn=0;nn<l;nn++){
                        int nn_one = games[nn*nr_class+i];
                        int nn_two = games[nn*nr_class+i+level];                    
                        if (nn_one > nn_two)
                            swap(nn_one, nn_two);
            
                        if (nn_one == one && nn_two == two){
                            C[m] = 
                                fabs(prob->cost[nn*nr_class+nn_one] - 
                                     prob->cost[nn*nr_class+nn_two]
                                    ) * param->C;
                            sub_prob.x[m] = prob->x[nn];
                            sub_prob.y[m] = (prob->cost[nn*nr_class+nn_one] < prob->cost[nn*nr_class+nn_two] ? -1 : +1);
                            idx[m] = nn;
                            if (C[m] > 0)
                                m++;
                        }
                    }
                    sub_prob.l = m;
                    f[pos] = svm_train_one(&sub_prob, param, C);
                    f[pos].idx = Malloc(int, sub_prob.l);
                    memcpy(f[pos].idx, idx, sizeof(int)*sub_prob.l);
                    f[pos].len = sub_prob.l;
                    
                    for(int nn=0;nn<l;nn++){
                        int nn_one = games[nn*nr_class+i];
                        int nn_two = games[nn*nr_class+i+level];                    
                        if (nn_one > nn_two)
                            swap(nn_one, nn_two);
                        if (nn_one != one || nn_two != two)
                            continue;
                        double res = -f[pos].rho;
                        for(int j=0;j<f[pos].len;j++){
                            if (f[pos].alpha[j] != 0.0)
                                res += f[pos].alpha[j] * Kernel::k_function(prob->x[nn], prob->x[f[pos].idx[j]], *param);
                        }
              
                        if (res > 0 || (fabs(res) == 0 && (rand() & 1)) || (isnan(res) && (rand() & 1))){
                            games[nn*nr_class+i] = two;
                            games[nn*nr_class+i+level] = one;
                        }
                        else{
                            games[nn*nr_class+i] = one;
                            games[nn*nr_class+i+level] = two;
                        }
                    }
                }//f[pos]       
            }
        }
        level *= 2;
    }

    free(sub_prob.x);
    free(sub_prob.y);

    for(int pos=0, i=0;i<nr_class;i++){
	for(int j=i+1;j<nr_class;j++){
	    for(int m=0;m<f[pos].len;m++){
                int n = f[pos].idx[m];
                if (prob->cost[n*nr_class+i] < prob->cost[n*nr_class+j])
                    nonzero[n*nr_class+i] |= 
                        (fabs(f[pos].alpha[m]) > 0);
                else
                    nonzero[n*nr_class+j] |= 
                        (fabs(f[pos].alpha[m]) > 0);
            }
            pos++;
        }
    }
    // build output
    model->nr_class = nr_class;
    
    model->label = label;
    for(int k=0;k<nr_class;k++)
        model->label[k]++;
    
    model->rho = Malloc(double,nr_class*(nr_class-1)/2);
    for(int p=0;p<nr_class*(nr_class-1)/2;p++)
        model->rho[p] = f[p].rho;
    
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
    
    int total_sv = 0;
    model->nSV = Malloc(int,nr_class);
    for(int k=0;k<nr_class;k++){
        int nSV = 0;                        
        for(int n=0;n<l;n++)
            if(nonzero[n*nr_class+k])
                ++nSV;
        
        model->nSV[k] = nSV;
        total_sv += nSV;
    }
    
    info("Total nSV = %d\n",total_sv);
    
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    idx = Malloc(int, l*nr_class);
    int *nz_start = Malloc(int,nr_class);
    for(int p=0, k=0;k<nr_class;k++){ //class first to group SV for each class together
        nz_start[k] = p;
        for(int n=0;n<l;n++){
            if(nonzero[n*nr_class+k]){
                model->SV[p] = prob->x[n];
                idx[n*nr_class+k] = p;
                p++;
            }
            else
                idx[n*nr_class+k] = -1;
        }
    }
    
    model->sv_coef = Malloc(double *,nr_class-1);
    for(int k=0;k<nr_class-1;k++){
        model->sv_coef[k] = Malloc(double,total_sv);
        memset(model->sv_coef[k], 0, sizeof(double)*total_sv);
    }
    
    for(int p=0, i=0;i<nr_class;i++){
        for(int j=i+1;j<nr_class;j++){
            // classifier (i,j): coefficients with
            // i are in sv_coef[j-1][nz_start[i]...],
            // j are in sv_coef[i][nz_start[j]...]
            for(int m=0;m<f[p].len;m++){
                int n = f[p].idx[m];
                if (nonzero[n*nr_class+i]){
                    int ref = idx[n*nr_class+i];
                    model->sv_coef[j-1][ref] = f[p].alpha[m];
                    f[p].alpha[m] = 0; //to avoid multiple inclusion
                }
            }
        
            for(int m=0;m<f[p].len;m++){
                int n = f[p].idx[m];                        
                if (nonzero[n*nr_class+j]){
                    int ref = idx[n*nr_class+j];
                    model->sv_coef[i][ref] = f[p].alpha[m];
                    f[p].alpha[m] = 0; //to avoid multiple inclusion
                }
            }
            ++p;
        }
    }
    free(C);
    free(idx);
    
    free(nonzero);
    for(int p=0;p<nr_class*(nr_class-1)/2;p++){
        free(f[p].idx);
        free(f[p].alpha);
    }
    free(f);
    free(nz_start);
}

void svm_predict_values_csapft(const svm_model *model, const svm_node *x, double* dec_values){
    return svm_predict_values_csovo_wap(model, x, dec_values);
}

double svm_predict_csapft(const svm_model *model, const svm_node *x){
    int i;
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
    svm_predict_values(model, x, dec_values);

    int *label = Malloc(int, nr_class);
    memcpy(label, model->label, sizeof(int)*nr_class);
    
    int level = 1;
    
    while(level <= nr_class){
	for(i=0;i+level<nr_class;i+=(level*2)){
	    int one = label[i]-1;
	    int two = label[i+level]-1;
	    int pos;
	    if (one < two){
		pos = (nr_class - 1 + nr_class - one + 1 - 1) * (one) / 2;
		pos += (two - one - 1);
		double res = dec_values[pos];
		if (res > 0 || (fabs(res) == 0 && (rand() & 1)) || (isnan(res) && (rand() & 1)))
		    label[i] = two+1;
		else
		    label[i] = one+1;
	    }
	    else{
		pos = (nr_class - 1 + nr_class - two + 1 - 1) * (two) / 2;
		pos += (one - two - 1);
		double res = dec_values[pos];
		if (res > 0 || (fabs(res) == 0 && (rand() & 1)) || (isnan(res) && (rand() & 1)))
		    label[i] = one+1;
		else
		    label[i] = two+1;
	    }
	}
	level*=2;
    }
    
    double result = label[0];
    free(label);
    return result;
}

svm_model *svm_load_model_csapft(FILE* fp, svm_model* model){
    return svm_load_model_csovo_wap(fp, model);
}

int svm_save_model_csapft(FILE* fp, const svm_model *model){
    return svm_save_model_csovo_wap(fp, model);
}
//====END OF CSAPTREE====

//====BEGIN OF CSSECOC====
inline int ceil_log2_cssecoc(int n){
    int order = 0;
    int len = n - 1;
    
    while(len != 0){
        order++;
        len >>= 1;
    }
  
    return order;
}

int **hada = NULL;
int hada_len = -1;

inline void free_hadamard_cssecoc(){
    for(int i=0;i<hada_len;i++)
        free(hada[i]);
    free(hada);
}

inline int** create_hadamard_cssecoc(int nr_class, int& len){
    int order = ceil_log2_cssecoc(nr_class);    
    len = (1 << order);
    
    if (hada != NULL){
	if (len == hada_len)
	    return hada;
	else
	    free_hadamard_cssecoc();
    }
    
    hada = Malloc(int*, len);
    for(int i=0;i<len;i++)
        hada[i] = Malloc(int, len);
    
    hada[0][0] = 1;
    
    for(int level = 1; level <= order; level++){
        int left = (1 << (level-1));
        for(int i = 0; i < left; i++){
            for(int r = 0; r < left; r++){
                for(int c = 0; c < left; c++){
                    hada[left+r][c] = hada[r][c];
                    hada[r][left+c] = hada[r][c];
                    hada[left+r][left+c] = -hada[r][c];
                }
            }
        }
    }    
    hada_len = len;
    return hada;
}


#define CSSECOC_NTHRES (6)

void svm_train_cssecoc(const svm_problem *prob, const svm_parameter *param, svm_model *model){
    int l = prob->l;
    int nr_class = prob->max_class;
    
    if(param->probability){
        fprintf(stderr, "probability output currently not supported in cost-sensitive setting\n");
        exit(-1);
    }
  
    int len;
    int** code = create_hadamard_cssecoc(nr_class, len);
  
    double *C = Malloc(double, l);
    int *idx = Malloc(int, l);
  
    int nT = CSSECOC_NTHRES;
  
    double *Ctotal = Malloc(double, l);
  
    for(int n=0;n<l;n++){
        double* pcost = &(prob->cost[n*nr_class]);
        Ctotal[n] = (*pcost);
        for(int i=1;i<nr_class;i++)
            Ctotal[n] += pcost[i];
    }
  
    int n_decision_function = 2 * len * nT;
    decision_function *f = Malloc(decision_function, n_decision_function);
  
    svm_problem sub_prob;
  
    sub_prob.x = Malloc(svm_node *,l);
    sub_prob.y = Malloc(double, l);
  
    for(int p=0, sign = +1; sign >= -1; sign -= 2){
        for(int i=0;i<len;i++){
            for(int tt=1;tt<=nT;tt++){
                double t = tt / (double)(nT+1.0);
                int m = 0;                          
                for(int n=0;n<l;n++){
                    double sum_cost = 0;
          
                    for(int j = 0; j < nr_class; j++){
                        if (code[i][j] * sign > 0)
                            sum_cost += prob->cost[n*nr_class+j];
                    }
          
                    C[m] = sum_cost - Ctotal[n] * t;
          
                    if (fabs(C[m]) > 0.0){
                        sub_prob.x[m] = prob->x[n];
                        sub_prob.y[m] = (C[m] >= 0 ? +1 : -1);
                        C[m] *= param->C;
                        C[m] = fabs(C[m]);                    
                        idx[m] = n;
                        m++;
                    }
                }
        
                sub_prob.l = m;
                f[p] = svm_train_one(&sub_prob,param,C);            
                f[p].len = m;
                f[p].idx = Malloc(int, f[p].len);
                memcpy(f[p].idx, idx, sizeof(int)*f[p].len);
                p++;                              
            }
        }
    }
    free(sub_prob.x);
    free(sub_prob.y);
    free(Ctotal);
    free(C);
    free(idx);

    // build output
    model->nr_class = nr_class;
  
    model->label = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++)
        model->label[i] = i+1;
  
    model->rho = Malloc(double,n_decision_function);
    for(int i=0;i<n_decision_function;i++)
        model->rho[i] = f[i].rho;
  
    {//no probability mode
        model->probA=NULL;
        model->probB=NULL;
    }
  
    int total_sv = 0;
    model->nSV = Malloc(int,n_decision_function);
    for(int i=0;i<n_decision_function;i++){
        int nSV = 0;                        
        for(int n=0;n<f[i].len;n++)
            if(fabs(f[i].alpha[n]) > 0)
                ++nSV;
    
        model->nSV[i] = nSV;
        total_sv += nSV;
    }
  
    info("Total nSV = %d\n",total_sv);
  
    model->l = total_sv;
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,n_decision_function);
  
    for(int p=0, i=0;i<n_decision_function;i++){
        model->sv_coef[i] = Malloc(double,model->nSV[i]);
        int m = 0;
        for(int n=0;n<f[i].len;n++){
            if(fabs(f[i].alpha[n]) > 0){
                model->SV[p++] = prob->x[f[i].idx[n]];
                model->sv_coef[i][m++] = f[i].alpha[n];
            }
        }
    }
    
    for(int i=0;i<n_decision_function;i++){
        free(f[i].idx);
        free(f[i].alpha);
    }
    free(f);
}

void svm_predict_values_cssecoc(const svm_model *model, const svm_node *x, double* dec_values){
    int i;
    int nr_class = model->nr_class;
    int len = (1 << ceil_log2_cssecoc(nr_class));
    int n_decision_function = 2 * len * CSSECOC_NTHRES;
  
    int start = 0;
    for(i=0;i<n_decision_function;i++){
        double res = -model->rho[i];
        for(int j=0;j<model->nSV[i];j++){
            res += model->sv_coef[i][j] * Kernel::k_function(x,model->SV[start+j],model->param);
        }
        dec_values[i] = res;
        start += model->nSV[i];
    }
}

double svm_predict_cssecoc(const svm_model *model, const svm_node *x){
    int nr_class = model->nr_class;
    int len;  
    int** code = create_hadamard_cssecoc(nr_class, len);
  
    int n_decision_function = 2 * len * CSSECOC_NTHRES;
    double *dec_values = Malloc(double, n_decision_function);
    svm_predict_values(model, x, dec_values);
  
    int *vote = Malloc(int,nr_class);
    for(int i=0;i<nr_class;i++)
        vote[i] = 0;
    int pos=0;
  
    for(int sign=+1;sign>=-1;sign-=2){
        for(int i=0;i<len;i++){
            for(int j=0;j<CSSECOC_NTHRES;j++){
                for(int k=0;k<nr_class;k++){
                    if (code[i][k]*sign*dec_values[pos] > 0)
                        --vote[k];
                }
                pos++;
            }
        }
    }

    int vote_max_idx = 0;
    for(int i=1;i<nr_class;i++){
        if(vote[i] > vote[vote_max_idx])
            vote_max_idx = i;
    }
    while(1){//a trick to randomly choose one when equal votes
	int i = rand() % nr_class;
	if (vote[i] == vote[vote_max_idx]){
	    vote_max_idx = i;
	    break;
	}
    }
    
    free(vote);
    free(dec_values);
    return model->label[vote_max_idx];
}

svm_model *svm_load_model_cssecoc(FILE* fp, svm_model* model){
    if(fp==NULL) return NULL;
  
    // read parameters
    svm_parameter& param = model->param;
  
    char cmd[81];
    while(1){
        fscanf(fp,"%80s",cmd);
    
        if(strcmp(cmd,"kernel_type")==0){
            fscanf(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++){
                if(strcmp(kernel_type_table[i],cmd)==0){
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL){
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            fscanf(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            fscanf(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            fscanf(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0){
            fscanf(fp,"%d",&model->nr_class);
	    int nr_class = model->nr_class;	    
	    int len = 1 << ceil_log2_cssecoc(nr_class);
	    int n_decision_function = 2 * len * CSSECOC_NTHRES;
	    
	    return svm_load_model_simple(fp, model, nr_class, n_decision_function);
	}
        else{
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }
    return NULL;
}

int svm_save_model_cssecoc(FILE* fp, const svm_model *model){
    if(fp==NULL) return -1;
    int nr_class = model->nr_class;
    int len = 1 << ceil_log2_cssecoc(nr_class);
    int n_decision_function = 2 * len * CSSECOC_NTHRES;
  
    return svm_save_model_simple(fp, model, nr_class, n_decision_function);
}
//====END OF CSSECOC====
