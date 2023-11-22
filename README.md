# VIEWS Prediction Comptetition: Bodentien & Rüter
This is the repository for all code related to participation (and bachelor thesis) in the 2023/24 VIEWS Prediction competition. 

All information regarding the competition can be found at https://viewsforecasting.org/prediction-competition-2/. 

## Model approaches
We model the predictive distribution of fatalities due to state-based conflicts on a country-month (cm) level using three different approaches. First, we utilize a **negative binomial distribution** whose parameters are estimated via empirical moments of the country's past $w$ fatalities to account for the **overdispersion** inherent in the data. Second, we employ a **hurdle model** that additionally accounts for zero-inflation by characterizing the distribution of zeros separately using a bernoulli variable. Positive numbers of fatalities are modeled via a truncated negative binomial distribution. Again, the respective model parameters are estimated based on past fatalities. Third, we flexibly incorporate additional feature variables provided by the ViEWS team using **feed-forward neural networks**. In all three cases, we choose the hyperparameters in such a way that the average continuous ranked probability score (**CRPS**) is minimized. We find that the simple negative binomial distribution outperforms the other two, more involved approaches. 

| Model                        | NB  | Hurdle | Neural Network NB | Neural Network |
|------------------------------|-----|--------|-------------------|----------------|
| Overdispersion               |  X  |   X    |        X          |       X        |
| Zero Inflation               |     |   X    |                   |       X        |
| Spatio-temporal dependencies |     |        |        X          |       X        |
| Complexity                   | low | middle |      middle       |      high      |


The complexity of the models increases from the low complexity NB model to the high complexity neural networks. Within the neural networks, similar to the baseline models, two models are created to predict conflict fatalities. The primary neural network model uses the energy form of the CRPS as its loss function. The additional model, referred to as neural network NB, models the parameters of the negative binomial distribution. This approach was added after evaluating the results of the baselines against the neural network model. The idea was to combine the best performing model, the NB model, with the neural network model as a more flexible modeling class utilizing the feature variables.