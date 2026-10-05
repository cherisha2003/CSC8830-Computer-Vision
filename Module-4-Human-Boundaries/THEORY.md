## Fourier theory: edges and regions

The assignment asks for **mathematical derivations**, not just a spectrum picture. This section connects the equations to the Fourier demonstration tab.

### 1. Representing an image by spatial frequencies

Let $f[x,y]$ be an $M\times N$ grayscale image, with $x$ the row and $y$ the column. The discrete Fourier transform (DFT) is

$$
F[u,v]=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f[x,y]\,e^{-j2\pi(ux/M+vy/N)}.
$$

The inverse transform reconstructs the pixels:

$$
f[x,y]=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F[u,v]\,e^{j2\pi(ux/M+vy/N)}.
$$

Each coefficient describes a repeating intensity pattern. The DC coefficient is the sum of image intensities. Low frequencies describe broad changes; higher frequencies describe rapid changes such as edges, fine textures, and noise. `fftshift` moves DC to the center for display. The app displays $\log(1+|F|)$ to make smaller coefficients visible; filtering still uses the full complex spectrum, including phase.

### 2. Deriving an edge operator in the frequency domain

For a continuous periodic image of width $W$ and height $H$, differentiate its Fourier-series reconstruction term by term:

$$
\frac{\partial}{\partial x}e^{j2\pi(kx/W+ly/H)}
=\frac{j2\pi k}{W}e^{j2\pi(kx/W+ly/H)}.
$$

Therefore, with frequencies $\nu_x=k/W$ and $\nu_y=l/H$,

$$
\mathcal{F}\{f_x\}=j2\pi\nu_x F,
\qquad \mathcal{F}\{f_y\}=j2\pi\nu_y F.
$$

Differentiation multiplies a coefficient by its frequency. Constant intensity has zero frequency, so its derivative is zero. Rapid changes receive larger weights.

For sampled images, a finite difference has its own exact transfer function. Define the circular forward difference $d_x[x,y]=f[(x+1)\bmod M,y]-f[x,y]$. Substitute $p=x+1$ in the DFT of the first term:

$$
\sum_{x,y}f[x+1,y]e^{-j2\pi(ux/M+vy/N)}
=e^{j2\pi u/M}\sum_{p,y}f[p,y]e^{-j2\pi(up/M+vy/N)}.
$$

Subtracting the transform of $f[x,y]$ gives

$$
D_x=(e^{j2\pi u/M}-1)F,\qquad
D_y=(e^{j2\pi v/N}-1)F.
$$

Using $e^{j\theta}-1=2j e^{j\theta/2}\sin(\theta/2)$,

$$
|e^{j2\pi u/M}-1|=2|\sin(\pi u/M)|.
$$

The response is zero at DC and large near Nyquist. Inverse-transform the two responses and combine them into edge strength:

$$
g_x=\operatorname{IDFT}(D_x),\quad g_y=\operatorname{IDFT}(D_y),\quad
E[x,y]=\sqrt{|g_x[x,y]|^2+|g_y[x,y]|^2}.
$$

An edge map can be obtained by thresholding $E$. This detects intensity changes, including texture; it does not identify humans by itself. Circular differences can introduce artificial edges between opposite image borders. Padding before filtering helps reduce border artifacts.

### 3. Deriving frequency filtering for region segmentation

Start with the convolution theorem. For circular convolution $s=f*h$,

$$
S[u,v]=H[u,v]F[u,v],\qquad s=\operatorname{IDFT}(HF).
$$

To see why, substitute $s[x,y]=\sum_{a,b}h[a,b]f[x-a,y-b]$ into its DFT and change variables to $p=x-a$, $q=y-b$. The exponential factors into a term in $(p,q)$ and a term in $(a,b)$; the two sums become $F$ and $H$.

For the app's centered spectrum $F_c$, define $D(u,v)$ as distance in frequency-bin coordinates from centered DC. An ideal low-pass filter with radius $R$ is

$$
H_L[u,v]=\begin{cases}1,&D(u,v)\le R,\\0,&D(u,v)>R.\end{cases}
$$

Undo the centering before applying the inverse DFT:

$$
s=\operatorname{IDFT}\{\operatorname{ifftshift}(H_LF_c)\},\qquad
r=\operatorname{IDFT}\{\operatorname{ifftshift}((1-H_L)F_c)\}=f-s.
$$

The low-pass reconstruction $s$ reduces fine texture. This can make pixels within a relatively uniform region easier to group. The residual $r$ emphasizes boundaries but also emphasizes unrelated texture. A hard cutoff can cause ringing; a Gaussian low-pass filter gives a smoother transition.

Frequency filtering must be followed by a region decision. For a bright target, one option is

$$
B[x,y]=\begin{cases}1,&s[x,y]>t^*,\\0,&s[x,y]\le t^*.\end{cases}
$$

To choose $t^*$ by Otsu's method, normalize the histogram into probabilities $p(i)$. Split gray levels at candidate $t$:

$$
\omega_0=\sum_{i\le t}p(i),\quad \omega_1=\sum_{i>t}p(i),\quad
\mu_0=\frac{\sum_{i\le t}i p(i)}{\omega_0},\quad
\mu_1=\frac{\sum_{i>t}i p(i)}{\omega_1}.
$$

Ignore thresholds with an empty class. Since $\mu_T=\omega_0\mu_0+\omega_1\mu_1$, the between-class variance simplifies to

$$
\sigma_B^2=\omega_0(\mu_0-\mu_T)^2+\omega_1(\mu_1-\mu_T)^2
=\omega_0\omega_1(\mu_0-\mu_1)^2.
$$

Thus

$$
t^*=\arg\max_t\sigma_B^2(t).
$$

Threshold the reconstruction, remove small regions, select the desired connected region using a prompt, and trace its contour. Thresholding is nonlinear; it is not another frequency filter. Filtering alone cannot assign a semantic label such as “human.” If foreground and background intensities overlap, this approach may fail.

**Connection to this project:** The Fourier tab demonstrates frequency filtering and a thresholded low-pass reconstruction. The thermal pipeline uses a spatial Gaussian blur followed by Otsu; a Gaussian blur is also low-pass filtering through the convolution theorem. The RGB pipeline uses seeded watershed, not Fourier filtering. These are related ideas, but they are separate implementations.

### 4. A small worked example

Take the circular row $f=[0,0,1,1]$. Its DFT is

$$
F=[2,-1+j,0,-1-j].
$$

Keeping only DC reconstructs $[0.5,0.5,0.5,0.5]$: the two regions have disappeared. This shows why excessive smoothing hurts segmentation. The forward difference is

$$
d=[0,1,0,-1].
$$

The $+1$ marks the rise into the bright region; the $-1$ includes the circular end-to-start transition. Its DFT is $[0,-2j,0,2j]$, exactly what $(e^{j2\pi u/4}-1)F[u]$ produces.

### What to explain during the demo

Reducing the low-pass radius smooths away more detail. Increasing it preserves finer detail. The high-pass view highlights rapid changes, while a threshold on the low-pass reconstruction groups pixels into regions. Neither view alone is a guaranteed human outline.
