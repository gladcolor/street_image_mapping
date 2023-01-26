# street_image_mapping

Still being developed, please wait!

Street Image Mapping (SIM) is a universal framework providing toolkit for localizing and measuring objects in street view images (SVI). SIM can automatically measure the street objects' 3D coordinates or size with appropriate parameters or auxiliary data. The current version of SIM provides two pipelines:
1) [tacheometric surveying](https://en.wikipedia.org/wiki/Tacheometry), i.e., localizing objects which have a known height or width (e.g., stop-sign).
2) width measuring for ribbon objects (e.g., road). 

These two pipelines can be used for vertical and horizontal measuring, respectively. More pipelines, such as triangulation, will be added in the future. 


Below is a notebook to show how to use this framework.


<a href="https://colab.research.google.com/drive/1sS1HmovMwxjax_0e8uqZm3Xtk-Xd-pef?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>
 
Figure 1 explains how the framework conducts tachometry. Panoramas are widely used in SVI services and are stored in equirectangular projection; each pixel's position (e.g., column and row number) represent its orientation. Thus, the street object's altitude angle and azimuth angle originating from the camera can be converted from its column and row number in the panorama. If the object’s distance from the panorama is known, its location can be computed. Our framework implemented the tacheometric method proposed by [Ning et al., (2021)](https://www.tandfonline.com/doi/abs/10.1080/13658816.2021.1981334) to obtain the object distance. 

$AB$ in Figure 1 denotes the object, and its height is knows as $h_o$. Also, Figure 1 illustrates the geometric relationship between $h_o$ the horizontal object distance $d_{hor}$  to the panorama camera $C$, where $\theta_t$ and $\theta_b$ are the altitude angle from $C$ to the object top and bottom respectively. 

The area of the triangle formed by sides $a$, $b$, and $h_o$ can be calculated by $h_o\cdot d_{hor}\cdot0.5$ or $sin\left(\theta_t+\theta_b\right)\cdot a\cdot b\cdot0.5$, so $h_o\cdot d_{hor}=sin\left(\theta_t+\theta_b\right)\cdot a\cdot b$; by plugging in $a=d_{hor}/cos\left(\theta_t\right)$ and $b=d_{hor}/cos\left(\theta_b\right)$, we can have Equation (1) to compute $d_{hor}$. The vertical distance of the object's bottom can be obtained by Equation (2). Therefore, the 3D coordinates of the target objects originating from the panorama camera $C$ can be obtained. For further localization of the target object, the panorama (i.e., $C$) 's coordinate is needed, which is usually provided in its metadata from SVI services.
$$d_{hor}=\frac{h_o⋅cosθ_t⋅cosθb}{sin(θ_t+θ_b)}    \qquad{       (1)}$$ 
$$h_b=tanθ_b·d_{hor}    \qquad{      (2)}$$


![img_2.png](doc_images/img_2.png)

Figure 1. Image-based tacheometric surveying
# Notes
[GeoPandas](https://geopandas.org/en/stable/) is required for SIM. If you use Windows and have difficulty to install `GeoPandas`, please refer to [this post](https://geoffboeing.com/2014/09/using-geopandas-windows/). Or using the following `conda` command to install:

`conda create -n geo --strict-channel-priority geopandas`
