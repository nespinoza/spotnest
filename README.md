# spotnest

SpotNest is a spot fitting routine (actually, a wrapper) which combines Bence Beky's *spot*rod (https://github.com/bencebeky/spotrod) and Multi*Nest* (via Johannes Buchner PyMultiNest, https://github.com/JohannesBuchner/PyMultiNest). It is a simple, ready-to-use code that is also easy to modify.

![SpotNest fit to data](spotnest.png?raw=true "Example of spotnest fit to data") 

Author: Néstor Espinoza (espinoza@mpia.de). 

This is a wrapper of the `spotrod` and `PyMultiNest` codes. If you use this code, please acknoledge also the authors of those packages. Citation for `spotrod`: http://adsabs.harvard.edu/abs/2014arXiv1407.4465B. Citation for `PyMultiNest`: http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1402.0004.

DEPENDENCIES
------------

This code makes use of three important libraries:

- **Numpy**.
- **spotrod** (https://github.com/bencebeky/spotrod).
- **PyMultiNest** (https://github.com/JohannesBuchner/PyMultiNest).

Optionally, you might install **Matplotlib** to see the final plots. 
All of them are open source and can be easily installed in any machine.

USAGE
-----

Once all of the above is installed, usage of the code is very easy. First, copy the `spotrod.so` file that you can generate using `spotrod` (https://github.com/bencebeky/spotrod) on this folder. After this, open the `spotnest.py` code, modify the parameters under `OPTIONAL INPUTS` and then you are good to go. An example mock dataset is given under `my_lightcurve.dat` to help you get started (with the current options and this dataset, you should be able to run the code as is to test it). Once everything is set, simply run:

    python spotnest.py

And the could should run, fit the transit + spot model, and save the results to a pickle file. If you have matplotlib, it will also show you a nice plot at the end with draws from the posterior model.

OUTPUTS
-------

The main output of the code are all the `PyMultiNest` related outputs, along with a pickle file with all the useful posterior information to make your life easier called `POSTERIOR_SAMPLES_nspot_x.pkl`, where `x` can be `0` for a no-spot transit fit, `1` for a transit plus one spot model and `2` for a transit plus two spots model. This pickle file is a dictionary, which contains:

    `posterior_samples`                     These are the posterior samples of the MultiNest posterior exploration. The first 
                                            This is an array of size `nsamples,nvariables`. The first six variables are always 
                                            the transit parameters fitted: rp (Rp/Rs), aR (a/Rs), inc (inclination of the orbit), 
                                            t0 (time of transit center), q1 (first transformed coefficient a-la-Kipping 2013), 
                                            q2 (same for the second transformed limb-darkening coefficient). The following four 
                                            variables are the spot parameters of the first spot: `x` and `y`, which are the 
                                            position of the spot on the stellar surface (in units of the stellar radii), the spot 
                                            radius (also in units of the stellar radii) and the spot contrast. There will be 
                                            additional four variables with the same parameters for the second spot if you fitted 
                                            two spots.

    `logZ`                                  One of the key features of MultiNest: this is the log-evidence of the model. Useful 
                                            for model comparison in order to decide if no spot model, a one-spot model or a 
                                            two-spot model is preferred by the data. 

EXAMPLE DATASET
---------------

The code is ready-to-use for the attached dataset under `my_lightcurve.dat`. If you run spotnest as-is, you should obtain a value of logZ of 633.18 for the model with no spot, 641.97 for the model with one spot and 641.13 for the model with two spots. This tells you that, assuming each of the models is equiprobable a-priori, the model with one spot is Z1/Z0 ~ 6530 times more likely than the model with no spot and Z1/Z2 ~ 2 times more likely than the model with two spots. The model with one spot is, thus, preferred by the data (and, actually, this was the model that generated the data!).
