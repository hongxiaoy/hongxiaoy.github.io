---
title: "Monocular Occupancy Prediction for Scalable Indoor Scenes"
collection: publications
permalink: /publication/2024-07-01-Monocular-Occupancy-Prediction-for-Scalable-Indoor-Scenes
excerpt: 'Camera-based 3D occupancy prediction has recently garnered increasing attention in outdoor driving scenes. However, research in indoor scenes remains relatively unexplored. The core differences in indoor scenes lie in the complexity of scene scale and the variance in object size. In this paper, we propose a novel method, named ISO, for predicting indoor scene occupancy using monocular images. ISO harnesses the advantages of a pretrained depth model to achieve accurate depth predictions. Furthermore, we introduce the Dual Feature Line of Sight Projection (D-FLoSP) module within ISO, which enhances the learning of 3D voxel features. To foster further research in this domain, we introduce Occ-ScanNet, a large-scale occupancy benchmark for indoor scenes. With a dataset size 40 times larger than the NYUv2 dataset, it facilitates future scalable research in indoor scene analysis. Experimental results on both NYUv2 and Occ-ScanNet demonstrate that our method achieves state-of-the-art performance.'
date: 2024-07-01
venue: 'The European Conference on Computer Vision (ECCV)'
paperurl: 'https://arxiv.org/pdf/2407.11730'
---

<span class="author-block">
<a href="https://orcid.org/0009-0003-9249-2726" target="_blank">Hongxiao Yu</a><sup>1,2</sup>,</span>
<span class="author-block">
<a href="https://orcid.org/0000-0002-6360-1431" target="_blank">Yuqi Wang</a><sup>1,2</sup>,</span>
<span class="author-block">
<a href="https://orcid.org/0000-0002-9555-1897" target="_blank">Yuntao Chen</a><sup>3</sup>,</span>
<span class="author-block">
<a href="https://orcid.org/0000-0003-2648-3875" target="_blank">Zhaoxiang Zhang</a><sup>1,2,3</sup></span>

<span class="author-block"><sup>1</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences (UCAS)</span><br>
<span class="author-block"><sup>2</sup>NLPR, MAIS, Institute of Automation, Chinese Academy of Sciences (CASIA)</span><br>
<span class="author-block"><sup>3</sup>Centre for Artificial Intelligence and Robotics (HKISI_CAS)</span><br>

<html>
<head>
  <meta charset="utf-8">
  <!-- Meta tags for social media banners, these should be filled in appropriatly as they are your "business card" -->
  <!-- Replace the content tag with appropriate information -->
  <meta name="description" content="DESCRIPTION META TAG">
  <meta property="og:title" content="SOCIAL MEDIA TITLE TAG"/>
  <meta property="og:description" content="SOCIAL MEDIA DESCRIPTION TAG TAG"/>
  <meta property="og:url" content="URL OF THE WEBSITE"/>
  <!-- Path to banner image, should be in the path listed below. Optimal dimenssions are 1200X630-->
  <meta property="og:image" content="static/image/your_banner_image.png" />
  <meta property="og:image:width" content="1200"/>
  <meta property="og:image:height" content="630"/>


  <meta name="twitter:title" content="TWITTER BANNER TITLE META TAG">
  <meta name="twitter:description" content="TWITTER BANNER DESCRIPTION META TAG">
  <!-- Path to banner image, should be in the path listed below. Optimal dimenssions are 1200X600-->
  <meta name="twitter:image" content="static/images/your_twitter_banner_image.png">
  <meta name="twitter:card" content="summary_large_image">
  <!-- Keywords for your paper to be indexed by-->
  <meta name="keywords" content="KEYWORDS SHOULD BE PLACED HERE">
  <meta name="viewport" content="width=device-width, initial-scale=1">


  <title>Academic Project Page</title>
  <link rel="icon" type="image/x-icon" href="static/images/favicon.ico">
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
  rel="stylesheet">

  <link rel="stylesheet" href="static/css/bulma.min.css">
  <link rel="stylesheet" href="static/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="static/css/bulma-slider.min.css">
  <link rel="stylesheet" href="static/css/fontawesome.all.min.css">
  <link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="static/css/index.css">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
  <script defer src="static/js/fontawesome.all.min.js"></script>
  <script src="static/js/bulma-carousel.min.js"></script>
  <script src="static/js/bulma-slider.min.js"></script>
  <script src="static/js/index.js"></script>
</head>
              

                  

                  <div class="column has-text-centered">
                    <div class="publication-links">
                         <!-- Arxiv PDF link -->
                      <span class="link-block">
                        <a href="https://arxiv.org/pdf/2407.11730" target="_blank"
                        class="external-link button is-normal is-rounded is-dark">
                        <span class="icon">
                          <i class="fas fa-file-pdf"></i>
                        </span>
                        <span>Paper</span>
                      </a>
                    </span>
                  <br>
                  <!-- Github link -->
                  <span class="link-block">
                    <a href="https://github.com/hongxiaoy/ISO" target="_blank"
                    class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                      <i class="fab fa-github"></i>
                    </span>
                    <span>Code</span>
                  </a>
                </span>
                <br>
                <!-- ArXiv abstract Link -->
                <span class="link-block">
                  <a href="https://arxiv.org/abs/2407.11730" target="_blank"
                  class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="ai ai-arxiv"></i>
                  </span>
                  <span>arXiv</span></a>


<!-- Paper abstract -->
<section class="section hero is-light">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
            Camera-based 3D occupancy prediction has recently garnered increasing attention in outdoor driving scenes. However, research in indoor scenes remains relatively unexplored. The core differences in indoor scenes lie in the complexity of scene scale and the variance in object size. In this paper, we propose a novel method, named ISO, for predicting indoor scene occupancy using monocular images. ISO harnesses the advantages of a pretrained depth model to achieve accurate depth predictions. Furthermore, we introduce the Dual Feature Line of Sight Projection (D-FLoSP) module within ISO, which enhances the learning of 3D voxel features. To foster further research in this domain, we introduce Occ-ScanNet, a large-scale occupancy benchmark for indoor scenes. With a dataset size 40 times larger than the NYUv2 dataset, it facilitates future scalable research in indoor scene analysis. Experimental results on both NYUv2 and Occ-ScanNet demonstrate that our method achieves state-of-the-art performance.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>
<!-- End paper abstract -->


<!--BibTex citation -->
  <section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
      <h2 class="title">BibTeX</h2>
      <pre><code>BibTex Code Here</code></pre>
    </div>
</section>
<!--End BibTex citation -->

