"""
Data structures for model_run
XM13: pretty pure

"""
ChannelData = u"""

    This table describes the REFERENCE ion channel densities (and voltage shifts if necessary)
    for different cell types based on the Xie and Manis 2013 models for mouse.

    The REFERENCE values are applied to "point" models, and to the soma of
    compartmental models.
    The names of the mechanisms must match a channel mechanism (Neuron .mod files)
    and the following _(gbar, vshift, etc) must match an attribute of that channel
    that can be accessed.

    -----------------------------------------------------------------------------------------------------------------------------------
                   II             II-I           I-c           I-II          I-t       
                                                                           
    nav11_gbar     1000.  [1]     1000.  [1]     3000.  [1]    1000.  [2]    3000.  [1] 
    kht_gbar       58.0   [3]     58.0   [1]     500.0  [1]    150.0  [2]    500.0  [1] 
    klt_gbar       80.0   [1]     14.0   [1]     0.0    [1]    20.0   [2]    0.0    [1] 
    ka_gbar        0.0    [1]     0.0    [1]     0.0    [1]    0.0    [2]    125.0    [1] 
    ihvcn_gbar     15.0   [1]     30.0   [1]     18.0   [1]    2.0    [2]    18.0   [1] 
    leak_gbar      2.0    [1]     2.0    [1]     8.0    [1]    2.0    [2]    8.0    [1] 
    leak_erev      -65    [1]     -65    [1]     -65    [1]    -65    [2]    -65    [1] 
    na_type        nav11  [1]     nav11  [1]     nav11  [1]    nav11  [1]    nav11  [1] 
    ih_type        ihvcn  [1]     ihvcn  [1]     ihvcn  [1]    ihvcn  [2]    ihvcn  [1] 
    soma_Cap       26.0   [1]     26.0   [1]     25.0   [1]    26.0   [2]    25.0   [1] 
    nav11_vshift   12.0    [1]     4.3    [1]     4.3    [1]    4.3    [1]    4.3    [1]
    e_k            -84    [1]     -84    [1]     -84    [1]    -84    [2]    -84    [1] 
    e_na           50.    [1]     50.    [1]     50.    [1]    50.    [2]    50.    [1] 
    ih_eh          -43    [1]     -43    [1]     -43    [1]    -43    [2]    -43    [1] 

    -----------------------------------------------------------------------------------------------------------------------------------

    [1] Uses channels from Rothman and Manis, 2003
        Conductances are for Mouse bushy cells
        Xie and Manis, 2013
        Age "adult", Temperature=34C
        Units are nS.
        nav11_vshift: was 4.3 in Xie and Manis (2013) for T-stellate cells; 0 for bushy cells
        ihvcn was 30, reduced to 15 to try to raise input resistance
        Here reset to 0 for bushy cells

    [2] Rothman and Manis, 2003, model I-II
        Some low-voltage K current, based on observations of
        a single spike near threshold and regular firing for higher
        currents (Xie and Manis, 2017)
        
    [3] Increased DR to force AP repolarization to be faster


    """

ChannelCompartments = u"""

    This table describes the ion channel densities relative to somatic densities,
    e.g., relative to REFERENCE densities in the table XM13_channels.
    and voltage shifts, for different compartments of the specified neuron,
    Conductances will be calculated from the Model derived from Xie and Manis 2013 for mouse
    (data table: XM13_channels).

    NOTE: unmyelinatedaxon and initialsegment are equivalent in George's models, but only "unmyelinatedaxon" is actually used.
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------
                   axon       unmyelinatedaxon     myelinatedaxon     initialsegment    hillock     soma        dendrite         primarydendrite    secondarydendrite

    nav11_gbar     3.0 [1]    1.0 [1]              1.0 [1]            1.0 [1]           5.0 [1]     0.5 [1]     0.5 [1]          0.25 [1]           0.25 [1]
    kht_gbar       1.0 [1]    1.0 [1]              0.01 [1]           2.0 [1]           1.0 [1]     1.0 [1]     1.0 [1]          0.5 [1]            0.25 [1]
    klt_gbar       1.0 [1]    1.0 [1]              0.01 [1]           1.0 [1]           1.0 [1]     1.0 [1]     0.5 [1]          0.5 [1]            0.25 [1]
    ihvcn_gbar     0.0 [1]    0.0 [1]              0.0 [1]            0.5 [1]           0.0 [1]     1.0 [1]     0.5 [1]          0.5 [1]            0.5 [1]
    leak_gbar      1.0 [1]    0.25 [1]             0.25e-3 [1]        1.0 [1]           1.0 [1]     1.0 [1]     0.5 [1]          0.5 [1]            0.5 [1]
    leak_erev      -65. [1]   -65. [1]             -65. [1]           -65. [1]          -65. [1]    -65. [1]    -65. [1]         -65. [1]           -65. [1]
    nav11_vshift   12.0  [2]   4.3 [2]              0.0 [2]            4.3 [2]           4.3 [2]     4.3 [2]     0.0  [2]         0.0  [2]            0.0 [2]
    na_type        nav11      nav11                nav11              nav11             nav11       nav11       nav11            nav11              nav11
    ih_type        ihvcn      ihvcn                ihvcn              ihvcn             ihvcn       ihvcn       ihvcn            ihvcn              ihvcn
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    [1] Scaling is relative to soma scaling. Numbers are estimates based on general distribution from literature on cortical neurons.
    [2] Set to 0 (was 4.3 in original model). Matches original Barela et al (2006) scaling.

    """
