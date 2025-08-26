#include "ipopt_c_api.h"
#include "IpStdCInterface.h"
#include <stdlib.h>

struct _ipopt_problem_t {
  IpoptProblem problem;
};

ipopt_problem_t *
ipopt_problem_create(int n, double *xL, double *xU, double m, double *gl, double *gu,
                     int nnzj, int nnzh, eval_f_cb eval_f,
                     eval_grad_f_cb eval_grad_f, eval_g_cb eval_g,
                     eval_jac_g_cb eval_jac_g, eval_h_cb eval_h) {
  IpoptProblem problem =
      CreateIpoptProblem(n, xL, xU, m, gl, gu, nnzj, nnzh, 0, eval_f, eval_g,
                         eval_grad_f, eval_jac_g, eval_h);
  ipopt_problem_t *ret = (ipopt_problem_t *)malloc(sizeof(ipopt_problem_t));
  ret->problem = problem;
  return ret;
}

void ipopt_problem_add_str_option(ipopt_problem_t *p, const char *param,
                                  const char *value) {
  if (p->problem != NULL) {
    AddIpoptStrOption(p->problem, (char *)param, (char *)value);
  }
}

void ipopt_problem_add_int_option(ipopt_problem_t *p, const char *param,
                                  int value) {
  if (p->problem != NULL) {
    AddIpoptIntOption(p->problem, (char *)param, value);
  }
}

void ipopt_problem_add_num_option(ipopt_problem_t *p, const char *param,
                                  double value) {
  if (p->problem != NULL) {
    AddIpoptNumOption(p->problem, (char *)param, value);
  }
}

void ipopt_problem_set_problem_scaling(ipopt_problem_t *p, double obj_scaling,
                                       double *x_scaling, double *g_scaling) {
  if (p->problem != NULL) {
    SetIpoptProblemScaling(p->problem, obj_scaling, x_scaling, g_scaling);
  }
}

enum ipopt_return_status ipopt_problem_solve(ipopt_problem_t *p, double *x,
                                             double *g, double *obj_val,
                                             double *mult_g, double *mult_x_L,
                                             double *mult_x_U, char *user_data) {
  if (p->problem != NULL) {
    enum ApplicationReturnStatus ret = IpoptSolve(
        p->problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U, user_data);
    return ret;
  }
  return Internal_Error;
}

void ipopt_problem_free(ipopt_problem_t *p) {
  if (p->problem != NULL) {
    FreeIpoptProblem(p->problem);
  }
  free(p);
}
