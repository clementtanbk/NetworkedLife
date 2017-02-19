import numpy as np


def matrix_to_tex(mat):
    # Convenience function for copying the matrix to latex
    for i in mat:
        print(' & '.join(str(_) for _ in i), end=' \\\\\n')


# Copied from tex document
s = r"""
0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
""".strip().replace('\\', '').replace('\n', ';').replace('&', ',')

A = np.array(np.matrix(s))
print('A', A, sep='\n')

C = A.T.dot(A)
print('C', C, sep='\n')

print('A^2', A.dot(A), sep='\n')
print('A^3', A.dot(A).dot(A), sep='\n')
