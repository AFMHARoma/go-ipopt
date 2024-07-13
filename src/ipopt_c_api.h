#ifndef GO_IPOPT_H_
#define GO_IPOPT_H_

#if defined(WIN32) || defined(WINDOWS) || defined(_WIN32) || defined(_WINDOWS)
#define IPOPTCAPICALL __declspec(dllexport)
#else
#define IPOPTCAPICALL
#endif

#endif