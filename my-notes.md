# OpenMP Tutorial - My Notes

## Intro to OpenMP

- OpenMP assumes SMP address space (all processors have equal cost to access al addresses). This is never really the case.

## Tasks

Tasks are added to a queue. Threads then work through the queue in parallel. The queue is filled by one thread whilst the others work through the task queue.

All work in OpenMP is a task. A parallel region is just an implicit task region.

```c
#pragma omp parallel // Create a team of threads 
{ 
  #pragma omp single // Only fill the task queue with ne processor
   { 
    p = listhead ;
    while (p) { 
       #pragma omp task firstprivate(p) // The single thread add the to queue here
       {         
         process (p);
       }
       p=next (p) ;
     } 
   } // Other threads wait here and work through tasks
}
```

### Task graphs 

Can create dependent tasks which OpenMP will handle

```c
int a,b,c;
#pragma omp parallel
{ 
  #pragma omp single nowait
   { 
      #pragma omp task depend(out: a)       
         taskA (&a);
      #pragma omp task depend(in: a)       
         taskB (a,&b);
      #pragma omp task 
         taskC (&c); 
   }   
}
```

# GPU

GPU support has existed since 4.0.

## GPU constructs

### target construct

To offload a code section onto GPU device use a `target` region.

```c
#pragma omp target [clause[[,]clause]...]
{...} // a structured block of code
```

#### Clauses 
- `if(scalar-expression)` – If the scalar-expression evaluates to false then the target region is executed by the host device in the host data environment.
- `device(integer-expression)` – The value of the integer-expression selects the device when a device other than the default device is desired.
- `private(list)  firstprivate(list)` – creates variables with the same name as those in the list on the device.  In the case of firstprivate, the value of the variable on the host is copied into the private variable created on the device.
- `map(map-type: list)` – map-type may be to, from, tofrom, or alloc.  The clause defines how the variables in list are moved between the host and the device. (Lots more on this later)...
- `nowait` – The target task is deferred which means the host can run code in parallel to the target region on the device.

### loop construct

shorthand for `#pragma omp teams distribute parallel for simd`

`#pragma omp [parallel] loop`
>shorthand for `#pragma omp teams distribute parallel for simd`
- The loop construct says that the iterations of the loop can be run in any order (concurrently)
- It’s a contract that says the loop does not contain:
    - OpenMP API calls
    - Calls to procedures with OpenMP directives
    - Any directive other than parallel/simd/loop
- The loop directive binds to the parallel (or teams) region it is found inside
- If not, it binds to the encountering thread, and the compiler can help out thanks to the “as-if” rule.
    - Descriptive parallelism.

#### Simple loop examples
Concurrent loops on the device
```c
int main(void) {
   int N = 1024;
   double A[N], B[N], C[N];

   #pragma omp target 
   #pragma omp loop
      for (int ii = 0; ii < N; ++ii) {
         C[ii] = A[ii] + B[ii];
      }
}
```

Reductions on device
```c
#include <omp.h>
#include <stdio.h>
static long num steps = 100000000;

int main() {
    double sum = 0.0;
    double step = 1.0 / ( double ) num steps ;

    // map(tofrom:sum) is required to copy reduced sum values back to the host
    #pragma omp target map(tofrom:sum)
    #pragma omp loop reduction (+:sum)
    for (int i=0; i<numsteps; i++){
        double x = (i + 0.5) ∗ step;
        sum += 4.0 / (1.0 + x ∗ x);
    }

    double pi = step ∗ sum;
    printf(” pi with %ld steps is %lf\n”, num steps, pi);
}
```

## Memory movement

All *statically allocated* data used within the target region is automatically mapped to and from the device at the start and end of the target region.

Scalar variables are, by default, copied as firstprivate to the device. 

Pointers are handles just like scalars but their data is not copied back.

### Mapping heap data

Data allocated on the heap (i.e. via malloc) must be explicitly copied between the device and host.
```c
int main(void) {
   int  ii=0, N = 1024;
   int* A = malloc(sizeof(int)*N);

   #pragma omp target map(A[0:N])
   {
     // N, ii and A all exist here
     // The data that A points to DOES exist here!
   }
}
```

The various forms of the map clause
- `map(to:list)`: On entering the region, variables in the list are initialized on the device using the original values from the host (host to device copy).
- `map(from:list)`:  At the end of the target region, the values from variables in the list are copied into the original variables on the host (device to host copy). On entering the region, the initial value of the variables on the device is not initialized.
- `map(tofrom:list)`: the effect of both a map-to and a map-from (host to device copy at start of region, device to host copy at end).
- `map(alloc:list)`: On entering the region, data is allocated and uninitialized on the device.
- `map(list)`: equivalent to `map(tofrom:list)`.

## Profiling GPU code 

use `nvprof`

## Efficient memory management

Memory should be mapped as few times as necessary. This can be achieved with `target data map`

```c
#pragma omp target data map(to:A[0:N], B[0:M]) map(from: C[0:P])
{
    #pragma omp target
    {do lots of stuff with A, B and C}
    
    {do something on the host (not with A,B,C)
    
    #pragma omp target
    {do lots of stuff with A, B, and C}
}
```

Data can be copied to the device within a target region after already being mapped if the `always` flag is used to get host updates.

```c
#pragma omp target data map(alloc: A[0:N])
{
    #pragma omp target map(always, tofrom: A[0:N])
    {...} // Use A 
    
    host_update(A);
    
    #pragma omp target map(always, tofrom: A[0:N])
    {...} // Use A 
}
```

To create task purely for transfering data to or from the device, we can use `target update`
```c
#pragma omp target data map(to: A[0:N],B[0:M]) map(from: C[0:P])
{
     #pragma omp target
           {do lots of stuff with A, B and C on the device}
     
     #pragma omp target update from(A[0:N])
     
     host_do_something_with(A)
     
     #pragma omp target update to(A[0:N])
     
     #pragma omp target
           {do lots more stuff with A, B, and C on the device}
}
```
### Target enter/exit data

We can be more flexible with defining target regions by using `target enter/exit`
```c
void init_array(int *A, int N) {
    for (int i = 0; i < N; ++i)
        A[i] = i;  
    #pragma omp target enter data map(to: A[0:N])
}
int main(void) {
    int N = 1024;
    int *A = malloc(sizeof(int) * N);
    
    init_array(A, N);
    
    #pragma omp target
    #pragma omp loop
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * A[i];
    
    #pragma omp target exit data map(from: A[0:N])
}
```

This can be optomised by making data transfer asynchronous and using dependencies with `nowait` and `depend`.

```c
void init_array(int *A, int N) {
    for (int i = 0; i < N; ++i) A[i] = i;
    // Any future target regions which rely on A will wait for this map to complete
    #pragma omp target enter data map(to: A[0:N]) nowait depend(out: A)
}

int main(void) {
    int N = 1024;
    int *A = malloc(sizeof(int) * N);

    init_array(A, N);
    
    #pragma omp target nowait depend(inout: A)
    #pragma omp loop
    for (int i = 0; i < N; ++i)
        A[i] = A[i] * A[i];
    
    #pragma omp taskwait

    #pragma omp target exit data map(from: A[0:N]) 
}
```

#### Benefits

One reason to use enter/exit over `{}` is to prevent issues from pointer swapping.

With `#pragma omp target data map(from: )` The from location is fixed from the start of the target data region. If pointers are swapped, data is still copied back to the original pointer

with `target exit data map(from: )`, if pointers are swapped, it will go to the new address.

## Efficient data structures 

- A structure of arrays tends to be more efficient than an array of structures.
- Access contiguous data to minimise cache invalidations.

## How to get more control than loop gives

`loop` doesn't give enough control when we need to break up work into smaller blocks.

### parallel for

parallel for works in a similar on a device as it does on the host
```c
#pragma omp target
#pragma omp parallel for
for (i=0;i<N;i++) 
    {...}
``` 

### teams and distribute

The `teams` construct
- Similar to the parallel construct
- It starts a league of teams
- Each team in the league starts with one initial thread – i.e. a team of one thread
- Threads in different teams cannot synchronize with each other
- The construct must be “perfectly” nested in a target construct

The `distribute` construct
- Similar to the for construct
- Loop iterations are workshared across the initial threads in a league
- No implicit barrier at the end of the construct
- dist_schedule(kind[, chunk_size])
    - If specified, scheduling kind must be static
    - Chunks are distributed in round-robin fashion in chunks of size chunk_size
    - If no chunk size specified, chunks are of (almost) equal size; each team receives at least one chunk

```c
#pragma omp target
#pragma omp teams
#pragma omp distribute
for (i=0;i<N;i++)
    {...}
```

### Combining teams, distribute and parallel for

Nested parallel teams
```c
#pragma omp target
#pragma omp teams distribute
for (i=0;i<N;i++)
    #pragma omp parallel for simd
    for (j=0;j<M;j++)
        {...}

```

Single loop which shows 
- 64 iterations assigned to 2 teams
- Each team has 4 threads
- Each thread has 2 SIMD lanes
```c
#pragma omp target teams distribute parallel for simd \
        num_teams(2) num_threads(4) simdlen(2) 
for (i=0; i<64; i++)
    {...}

```
