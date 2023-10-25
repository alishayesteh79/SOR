import numpy as np
import math
import time

start =  time.time()

# matrix A
n = 2000
A = np.zeros(shape=(n,n))

for i in range(n):
    temp_vec = []

    for j in range(i-1):
        temp_vec.append(0)
    
    if i != 0 :
        temp_vec.append(1)

    temp_vec.append(3)

    if i != n-1 :
        temp_vec.append(1)

    for j in range(n-i-2):
        temp_vec.append(0)

    A[i] = temp_vec


# matrix b :
b = np.zeros(shape=(n,1))

for i in range(n):
    b[i][0] = i+1


# calculate w optimum

#   1) generate jacobi iterative matrix 
D_inv = np.zeros(shape=(n,n))
for i in range(n):
    D_inv[i][i] = 1/3

l = np.zeros(shape=(n,n))
for i in range(1,n):
    for j in range(0,i):
        l[i][j] = A[i][j]

u = np.zeros(shape=(n,n))
for i in range(0,n-1):
    for j in range(i+1, n):
        u[i][j] = A[i][j]

# 2) calculate mj
M = np.matmul((-D_inv),(l+u))

# 3) M eigen values
v,v1 = np.linalg.eig(M)
# find maximum of eigen values
m1 = max(v)
m2 = min(v)
if abs(m2) > m1 :
    mx = abs(m2)
else :
    mx = m1

# caculate w 
w = 2/(1 + math.sqrt(1-(mx**2)))


#primary guess
x = np.zeros(shape=(n,1))

x_new = x
e = 1000

while(e > 5e-14):
    for i in range(n):

        temp1 = 0
        for j in range(i):
            temp1 += A[i][j]*x_new[j]

        temp2 = 0 
        for j in range(i,n):
            temp2 += A[i][j]*x[j]

        x_new[i] = x[i] + (w/A[i][i])*(b[i] - (temp1 + temp2) )

    #calculate error
    #   infinite norm of b is 2000
    e = np.linalg.norm((b - (np.matmul(A,x_new))), np.inf)/2000

end = time.time()

with np.printoptions(threshold=np.inf):
    print(x_new)
print("\ntime: ", (end-start)*1000)