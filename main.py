""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.integrate import quad

def task1():
    # probability density functions with change of variables, check that you obtain a valid transformed pdf
    
    """ Start of your code
    """
    def p_z(z):
        if z <= 0:
            return 0
        return (1 / np.sqrt(2 * np.pi * z)) * np.exp(-0.5 * z)

    integral_result, error_estimate = quad(p_z, 0, np.inf)
    print("Integral result: ", integral_result)

    """ End of your code
    """

def task2(x, K):
    """ Multivariate GMM

        Requirements for the plots: 
        fig1
            - ax[0,k] plot the mean of each k GMM component, the subtitle contains the weight of each GMM component
            - ax[1,k] plot the covariance of each k GMM component
        fig2: 
            - plot the 8 samples that were sampled from the fitted GMM
    """
    
    mu, sigma, pi = [], [], np.zeros((K)) # modify this later
    num_samples = 10

    fig1, ax1 = plt.subplots(2, K, figsize=(2*K,4))
    fig1.suptitle('Task 2 - GMM components', fontsize=16)

    fig2, ax2 = plt.subplots(2, num_samples//2, figsize=(2*num_samples//2,4))
    fig2.suptitle('Task 2 - samples', fontsize=16)

    """ Start of your code
    """
    ### TASK 1.2

    M = np.shape(x)[1]
    D = M*M
    mu = np.random.rand(K, D)
    pi = np.random.rand(K)
    sigma = np.array([np.identity(D) for k in range(K)])
    # return (mu, sigma, pi), (fig1,fig2)
    print(f"{np.shape(pi)=}, {np.shape(sigma)=}")

    ### TASK 1.3
    samples = np.random.choice(range(np.shape(x)[0]), size=K, replace=False)
    mu = np.array([np.reshape(x[sample], D) for sample in samples])

    print(f"{np.shape(mu)=}")
    
    eps = 1e-4
    J = 100
    for j in range(J):
        assignments = np.array([np.argmin([np.linalg.norm(np.reshape(x_i, D) - mu_k) for mu_k in mu]) for x_i in x])
        new_mu = np.array([np.reshape(np.mean(x[assignments == k], axis=0), D) for k in range(K)])
        plt.imshow(np.reshape(mu[0,:], [M,M]))
        if np.allclose(mu, new_mu, atol=eps):
            break
        mu = new_mu
    print(j)
    """ End of your code
    """

    for k in range(K):
        ax1[0,k].set_title('C%i with %.2f' %(k,pi[k])), ax1[0,k].axis('off'), ax1[1,k].axis('off')

    return (mu, sigma, pi), (fig1,fig2)

def task3(x, mask, m_params):
    """ Conditional GMM

        Requirements for the plots: 
        fig
            - ax[s,0] plot the corrupted test sample s
            - ax[s,1] plot the restored test sample s (by using the posterior expectation)
            - ax[s,2] plot the groundtruth test sample s 
    """
    
    S, sz, _ = x.shape

    fig, ax = plt.subplots(S,3,figsize=(3,8))
    fig.suptitle('Task 3 - Conditional GMM', fontsize=12)
    for a in ax.reshape(-1):
        a.axis('off')
        
    ax[0,0].set_title('Condition',fontsize=8), ax[0,1].set_title('Posterior Exp.',fontsize=8), ax[0,2].set_title('Groundtruth',fontsize=8)
    for s in range(S):
        ax[s,2].imshow(x[s], vmin=0, vmax=1., cmap='gray')

    """ Start of your code
    """
    M = np.shape(x)[1]
    D = M*M
    mask = np.zeros(D)
    samples = np.random.choice(range(D), size=round(D/10), replace=False)
    for sample in samples:
        mask[sample] = 1
    
    x_corrupted = x.copy()
    for k in range(S):
        x_corrupted[k] = np.reshape(x_corrupted[k].flatten()*mask, [M, M])
    


    """ End of your code
    """

    return fig

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # Task 1: transformations of pdfs
    task1()

    # load train and test data
    with np.load("data.npz") as f:
        x_train = f["train_data"]
        x_test = f["test_data"]

    # Task 2: fit GMM to FashionMNIST subset
    K = 6 # TODO: adapt the number of GMM components
    gmm_params, fig1 = task2(x_train,K)

    # Task 2: inpainting with conditional GMM
    mask = None
    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    
