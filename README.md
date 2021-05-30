# Stock Price Prediction Uing Deep Learning Method

#### Investigating some attributes of deep learning methods when apply to stock price prediction problem
LSTM and GRU are usually the common approaches to predict future stock price on historical stock price.

There are four different companies' stock price from the 1980s to 2010s:
- AAPL (Apple)
- FB (Facebook)
- TSLA (Tesla)
- MSFT (Microsoft)

Some result's images:
![Stock price prediction on Facebook using GRU32](/graphs/FB_GRU32.png)
![Stock price prediction on Facebook using LSTM 32](graphs/FB_LSTM32.png)
![Stock price prediction on Facebook using GRU128](graphs/FB_GRU128.png)
![Stock price prediction on Facebook using LSTM128](graphs/FB_LSTM128.png)

Some results when apply model trained on one company to another company:
![Stock price prediction model trained on Facebook's stock price using GRU32when being applied on Tesla company](/graphs/FB_TSLA_GRU32.png)
![Stock price prediction model trained on Facebook's stock price using GRU32when being applied on Tesla company](/graphs/FB_TSLA_LSTM32.png)
![Stock price prediction model trained on Facebook's stock price using GRU32when being applied on Tesla company](/graphs/FB_TSLA_GRU128.png)
![Stock price prediction model trained on Facebook's stock price using GRU32when being applied on Tesla company](/graphs/FB_TSLA_LSTM128.png)
