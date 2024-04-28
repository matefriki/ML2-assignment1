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
from tqdm import tqdm
import scipy

from scipy.integrate import quad
from scipy.stats import multivariate_normal


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
    
    mu, sigma, pi = [], [], np.zeros((K))  # modify this later
    num_samples = 10

    fig1, ax1 = plt.subplots(2, K, figsize=(2*K, 4))
    fig1.suptitle('Task 2 - GMM components', fontsize=16)

    fig2, ax2 = plt.subplots(2, num_samples//2, figsize=(2*num_samples//2, 4))
    fig2.suptitle('Task 2 - samples', fontsize=16)

    """ Start of your code
    """

    np.random.seed(42)
    ### TASK 1.2
    # print(f"{np.shape(x)=}")
    # x = x[:2000, :, :]
    print(f"{np.shape(x)=}")
    M = np.shape(x)[1]
    D = M*M
    S = np.shape(x)[0]
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
    print(f"Finished k-means with {j=}")

    ### TASK 1.4
    # Compute the log likelihood

    def compute_cholesky_decomp(sigma_in):
        cholesky = np.empty_like(sigma_in)
        # Compute the Cholesky decomposition for each k
        for k in range(K):
            cholesky[k] = np.linalg.cholesky(sigma_in[k] + (1e-4)*np.eye(D))
        log_det_sigma_half = np.zeros(K)
        for k in range(K):
            # det_sigma_half[k] = np.prod(np.diag(cholesky[k]))
            log_det_sigma_half[k] = np.sum(np.log(np.diag(cholesky[k])))
        # print("\n**************\nCholesky:")
        # print(np.diag(cholesky[0]))
        # print("\n**************\nEnd Cholesky\n")
        return cholesky, log_det_sigma_half

    def logsumexp_stable(weights, exps):
        e_max = np.max(exps)
        return e_max + np.log(np.sum(np.exp(exps - e_max)))


    def log_likelihood(x, mu, sigma, pi, K, cholesky, log_det_sigma_half, exps):
        ll = 0
        llarray = np.zeros(S)
        for s in tqdm(range(S)):
            llarray[s] = logsumexp_stable(pi, exps=exps[s] - 0.5*log_det_sigma_half )
        return np.sum(llarray)


    # EM algorithm
    eps = 1e-4
    J = 100
    ll_old = 0
    exps = np.zeros((S, K))
    print(f"{np.shape(exps)=}")
    responsibilities = np.zeros((S,K))
    stopping_criterion = False
    print(f"{ll_old=}")
    # print(f"{ll_gab=}")


    for j in range(J):
        print(f"hola {j}")
        # pre-computations
        cholesky, log_det_sigma_half = compute_cholesky_decomp(sigma)

        for s in range(S):
            for k in range(K):
                # When computing (x-mu)Sigma-1(x-mu), it is equal to (x-mu)^T(L*L^T)^-1(x-mu) = (L^-1(x-mu))^T*(L^-1(x-mu))
                # Making a change of variables aux = L^-1(x-mu), then the exp is equal to aux^T*aux
                # computing aux with scipy is much faster, but scipy may not be allowed in the assignment
                # comment/uncomment on your convenience

                # aux = np.linalg.solve(cholesky[k], x[s].flatten() - mu[k])
                aux = scipy.linalg.cho_solve((cholesky[k], True), x[s].flatten() - mu[k])
                exps[s,k] = (-0.5)*np.dot(aux,aux)


        ll_new = log_likelihood(x, mu, sigma, pi, K, cholesky, log_det_sigma_half, exps)

        if j > 0:
            print(f"{ll_new=}, {ll_old=}")
            if np.abs(ll_old - ll_new) < eps:
                break
        ll_old = ll_new
        # compute responsibilities
        # for s in range(S):
        #     for k in range(K):
        #         responsibilities[s, k] = (pi[k]/det_sigma_half[k])*np.exp(exps[s,k])

        # responsibilities = (pi)*np.exp(exps-0.5*log_det_sigma_half)

        logres = np.log(pi) + exps-0.5*log_det_sigma_half
        for s in range(S):
            logres[s,:] = logres[s,:] - np.max(logres[s,:])
            responsibilities[s,:] = np.exp(logres[s,:])
            thesum = np.sum(responsibilities[s,:])
            responsibilities[s,:] = responsibilities[s,:]/thesum

        # print("previous logres")
        # print(logres)
        # logres = logres - np.max(logres)
        # responsibilities = np.exp(logres)

        # print("Previous responsibilities")
        # print(responsibilities)
        # print("logres")
        # print(logres)



        # # for s in range(S):
        # #     thesum = np.sum(responsibilities[s,:])
        # #     responsibilities[s, :] = responsibilities[s,:]/thesum
        # thesum = np.sum(responsibilities, axis=1)  # Sum along each row
        # responsibilities = responsibilities / thesum[:, np.newaxis]  # Divide each row element-wise by the sum

        print("Check Nans")
        if np.isnan(responsibilities).any():
            print("pi: ")
            print(pi)
            print("det_sigma_half")
            print(log_det_sigma_half)
            print("exps")
            print(exps)
            print("the_sum")
            print(thesum)
            print("responsibilities")
            print(responsibilities)
            break
        # print(np.isnan(responsibilities).any())

        N = np.sum(responsibilities, axis = 0)

        for k in range(K):
            # aux = np.zeros((S, M, M))
            # for s in range(S):
            #     for m1 in range(M):
            #         for m2 in range(M):
            #             aux[s,m1,m2] = responsibilities[s, k]*x[s,m1,m2]
            aux = responsibilities[:, k][:, np.newaxis, np.newaxis] * x
            mu[k] = np.sum(aux, axis=0).flatten()/N[k]

        # # x has shape (S, D)
        # # mu has shape (D, )
        # # w has shape (S, )
        # xmu = np.zeros((S, D))
        # for s in range(S):
        #     for d in range(D):
        #         xmu[s,d] = x[s,d] - mu[d]
        # bigmat = np.zeros((S, D, D))
        # for s in range(S):
        #     for d1 in range(D):
        #         for d2 in range(D):
        #             bigmat[s, d1, d2] = w[s]*xmu[s, d1, d2]*xmu[s, d2, d1]

        for k in range(K):
            xmu = x.reshape([S,D]) - mu[k]
            # Calculate outer products for each sample and scale by w
            # First, calculate the outer product for each 's' without the weights
            bigmat = np.einsum('si,sj->sij', xmu, xmu)

            # Now, multiply by w, which is reshaped to (S, 1, 1) for broadcasting
            bigmat *= responsibilities[:, k][:, np.newaxis, np.newaxis]
            sigma[k] = np.sum(bigmat, axis=0)/N[k]

        pi = N/S

    print(f"Finished with {j=}")

    for k in range(K):
        ax1[0, k].imshow(mu[k].reshape([M, M]))


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
            - ax[s,2] plot the groundtruth test sample s ^12
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
    mu, sigma, pi = m_params

    def compute_conditional_distribution(m_params, x_2, mask):
        mu, sigma, pi = m_params
        K = mu.shape[0]
        D = mu.shape[1]

        mu_cond = np.zeros((K, D))
        sigma_cond = np.zeros((K, D, D))
        pi_cond = np.zeros(K)
        log_pi_cond = np.zeros(K)

        mask = mask.astype(bool)
        for k in range(K):
            # Partition the mean and covariance matrix
            mu_1 = mu[k][~mask]
            mu_2 = mu[k][mask]
            sigma_11 = sigma[k][np.ix_(~mask, ~mask)]
            sigma_22 = sigma[k][np.ix_(mask, mask)]
            sigma_12 = sigma[k][np.ix_(~mask, mask)]
            sigma_21 = sigma[k][np.ix_(mask, ~mask)]

            # Regularize sigma_22 to avoid singularity
            regularization_term = 1e-4 * np.eye(sigma_22.shape[0])
            sigma_22_reg = sigma_22 + regularization_term

            # Compute conditional distribution
            # sigma_22_inv = np.linalg.inv(sigma_22_reg)
            # mu_cond[k][~mask] = mu_1 + sigma_12 @ sigma_22_inv @ (x_2 - mu_2)
            # sigma_cond[k][np.ix_(~mask, ~mask)] = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21

            mu_cond[k][~mask] = mu_1 + sigma_12 @ np.linalg.solve(sigma_22_reg, x_2 - mu_2)
            sigma_cond[k][np.ix_(~mask, ~mask)] = sigma_11 - sigma_12 @ np.linalg.solve(sigma_22_reg, sigma_21)

        for k in range(K):
            cholesky = np.linalg.cholesky(sigma_22_reg)
            log_pi_cond[k] = np.log(pi[k]) -0.5*(np.sum(np.log(np.diag(cholesky)))) - 0.5*(x_2 - mu_2) @ np.linalg.solve(sigma_22_reg, (x_2 - mu_2))
        log_pi_cond = log_pi_cond - np.max(log_pi_cond)
        pi_cond = np.exp(log_pi_cond)
        thesum = np.sum(pi_cond)
        pi_cond = pi_cond / thesum

        return mu_cond, sigma_cond, pi_cond

    M = np.shape(x)[1]
    D = M*M
    mask = np.zeros(D)
    samples = np.random.choice(range(D), size=round(D/10), replace=False)
    for sample in samples:
        mask[sample] = 1

    x_flattened = x.reshape(S, -1)
    mask_flattened = mask.flatten()
    mask_flattened_boolean = mask_flattened.astype(bool)

    x_restored = np.empty_like(x_flattened)
    x_corrupted = np.empty_like(x_flattened)
    for s in range(S):

        x_corrupted[s] = x_flattened[s] * mask_flattened_boolean
        x_2 = x_flattened[s][mask_flattened_boolean]
        mu_cond, sigma_cond, pi_cond = compute_conditional_distribution(m_params, x_2, mask_flattened)

        # Compute the expected value for the missing pixels, weighted by the component probabilities
        # Placeholder for computing the actual expectation
        # print(f"{pi=}, {pi_cond=}")
        x_m_expected = np.sum([pi_cond[k] * mu_cond[k] for k in range(K)], axis=0)

        # Fill in the observed and expected missing pixels to restore the image
        x_restored[s][mask_flattened_boolean] = x_2
        x_restored[s][~mask_flattened_boolean] = x_m_expected[~mask_flattened_boolean]

        # Reshape the restored data back into image format and display
        ax[s, 1].imshow(x_restored[s].reshape(M, M), vmin=0, vmax=1., cmap='gray')
        # Corrupt the original image with the mask and display
        ax[s, 0].imshow(x_corrupted[s].reshape(M, M), vmin=0, vmax=1., cmap='gray')

    # plt.tight_layout()
    # plt.show()
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
    K = 10 # TODO: adapt the number of GMM components
    gmm_params, fig1 = task2(x_train,K)

    # Task 2: inpainting with conditional GMM
    mask = None
    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    
