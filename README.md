# Music Retrieval System using Query-by-Humming/Singing
Author: Laia Albors

Director: José Adrián Rodriguez Fonollosa

Final Thesis

BSc Data Science and Engineering

### Description
Nowadays there are many ways to find a song through devices: searching for part of the lyrics, the name of the artist, the title of the song/album, among others; that is to say, using a search-by-text. There are also some applications that allow you to get the name of a song by recording it when it is playing in the background. The limitation of these methods is when the user does not remember the lyrics or has not had time to record the song while it was playing in the background. A possible solution to this difficulty is for the user to sing/hum the song he/she wants to search for. This idea, however, is accompanied by a lot of challenges and problems, mainly related to the variability of the input: as not everyone has musical training - or simply because the user does not remember the song perfectly - it may happen that, when singing/humming it, he/she sings it in a different key, with a different speed, changes some rhythms, goes out of tune, etc. In the literature we can find several researches that face this challenge, but most of them use very basic and limited techniques. In this work, therefore, we propose to use more advanced techniques, based on deep learning, to improve the results obtained with traditional methods.

Thus, the objective of this Final Thesis is to develop a model capable of identifying a song from the user singing or humming it, using deep learning techniques.

### Methodology

1.    Obtain the Data Base: Justin Salamon, J. Serrà i Emilia Gómez. “Tonal Representations for Music Retrieval: From Version Identification to Query-by-Humming”. A: International Journal of Multimedia Information Retrieval, special issue on Hybrid Music Information Retrieval 2 (2013), pàg. 45 - 58. doi: http://dx.doi.org/10.1007/s13735-012-0026-0. url: http://hdl.handle.net/10230/41911

2.    Preprocess:

    a. Extract vocal part
    
    b. Data Augmentation
    
    c. Delete silence
    
    d. Segment the audios

3.    Extract features from the recordings

    a. MELODIA: Justin Salamon i Emilia Gómez. “Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics”. A: IEEE Transactions on Audio, Speech and Language Processing 20 (2012), pàg. 1759 - 1770. url: http://hdl.handle.net/10230/42183.
    
    b. YAAPT: Stephen Zahorian i Hongbing Hu. “A spectral/temporal method for robust fundamental frequency tracking”. A: The Journal of the Acoustical Society of America 123 (jul. de 2008), pàg. 4559 - 71. doi: 10.1121/1.2916590.
    
    c. VGGish: Shawn Hershey et al. “CNN Architectures for Large-Scale Audio Classification”. A: CoRR abs/1609.09430 (2016). arXiv: 1609.09430. url: http://arxiv.org/abs/1609.09430.

4.    Triplet Network

5. KNN