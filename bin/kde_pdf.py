import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate random sample from normal distribution
sample = np.random.normal(loc=0, scale=1, size=1000)

# Calculate KDE
kde = stats.gaussian_kde(sample)

# Evaluate KDE and PDF at a range of x-values
x = np.linspace(-5, 5, num=100)
kde_vals = kde.evaluate(x)
pdf_vals = stats.norm.pdf(x, loc=0, scale=1)

# Plot results
fig, ax = plt.subplots()
ax.plot(x, kde_vals, label='KDE')
ax.plot(x, pdf_vals, label='PDF')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.legend()
# plt.show()
plt.savefig('/media/jorrit/Storage SSD/fathomnet/export/kde_pdf.png')
