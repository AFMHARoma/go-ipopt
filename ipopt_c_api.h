#ifndef GO_IPOPT_H_
#define GO_IPOPT_H_

#include <stdbool.h>

#if defined(WIN32) || defined(WINDOWS) || defined(_WIN32) || defined(_WINDOWS)
#define IPOPTCAPICALL __declspec(dllexport)
#else
#define IPOPTCAPICALL
#endif

enum ipopt_return_status {
  solve_succeeded = 0,
  solved_to_acceptable_level = 1,
  infeasible_problem_detected = 2,
  search_direction_becomes_too_small = 3,
  diverging_iterates = 4,
  user_requested_stop = 5,
  feasible_point_found = 6,

  maximum_iterations_exceeded = -1,
  restoration_failed = -2,
  error_in_step_computation = -3,
  maximum_cputime_exceeded = -4,
  maximum_walltime_exceeded = -5, ///< @since 3.14.0

  not_enough_degrees_of_freedom = -10,
  invalid_problem_definition = -11,
  invalid_option = -12,
  invalid_number_detected = -13,

  unrecoverable_exception = -100,
  non_ipopt_exception_thrown = -101,
  insufficient_memory = -102,
  internal_error = -199
};

typedef bool (*eval_f_cb)(int n, double *x, bool new_x, double *obj_value,
                          void *user_data);
typedef bool (*eval_grad_f_cb)(int n, double *x, bool new_x, double *grad_f,
                               void *user_data);

typedef bool (*eval_g_cb)(int n, double *x, bool new_x, int m, double *g,
                          void *user_data);

typedef bool (*eval_jac_g_cb)(int n, double *x, bool new_x, int m, int nele_jac,
                              int *iRow, int *jCol, double *values,
                              void *user_data);
typedef bool (*eval_h_cb)(int n, double *x, bool new_x, double obj_factor, int m,
                          double *lambda, bool new_lambda, int nele_hess,
                          int *iRow, int *jCol, double *values, void *user_data);

typedef struct _ipopt_problem_t ipopt_problem_t;

IPOPTCAPICALL ipopt_problem_t *
ipopt_problem_create(int n, double *xL, double *xU, int m, double *gl, double *gu,
                     int nnzj, int nnzh, eval_f_cb eval_f,
                     eval_grad_f_cb eval_grad_f, eval_g_cb eval_g,
                     eval_jac_g_cb eval_jac_g, eval_h_cb eval_h);
IPOPTCAPICALL void ipopt_problem_add_str_option(ipopt_problem_t *p,
                                                const char *param,
                                                const char *value);
IPOPTCAPICALL void ipopt_problem_add_int_option(ipopt_problem_t *p,
                                                const char *param, int value);
IPOPTCAPICALL void ipopt_problem_add_num_option(ipopt_problem_t *p,
                                                const char *param, float value);
IPOPTCAPICALL void ipopt_problem_set_problem_scaling(ipopt_problem_t *p,
                                                     float obj_scaling,
                                                     float *x_scaling,
                                                     float *g_scaling);
IPOPTCAPICALL enum ipopt_return_status
ipopt_problem_solve(ipopt_problem_t *p, double *x, double *g, double *obj_val,
                    double *mult_g, double *mult_x_L, double *mult_x_U,
                    char *user_data);
IPOPTCAPICALL void ipopt_problem_free(ipopt_problem_t *p);

#endif