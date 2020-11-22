# Which Marketing Channel Gives You the Highest Value Customers?


## CAC is overrated. Long live LTV.

> No one says, “I want the cheapest lawyers.” - [Peter Fader](https://marketing.wharton.upenn.edu/profile/faderp/)


So why go after the cheapest customers? But that’s precisely what a marketer does when they look at each channel’s CAC and then throw more money at the channel with the lowest CAC. Not all channels are created equally. Not all customers are created equally. 

Let’s return to the dataset provided by Prof Fader’s colleague, Daniel McCarthy. We’re going to figure out where our best customers come from. 

**We want the highest value customers, not the cheapest customers.**

### Roadmap 

1. Install the lifetimes library
2. Load in your customer + transactional data
3. Model number of purchases
4. Verify assumptions are met for gamma-gamma model
5. Model customer value 
6. Describe customer value by channel

### Installing Library


Let’s install and load the lifetimes library into our Jupyter Notebook or Colab instance. 

Playing along at home with your own data? You’ll want your data to look like this:

Taken all together, your notebook should look like this

```python
!pip install lifetimes

import pandas as pd

df = pd.read_csv(path)
df.head()
```


### Preparing the Data


Before we attempt to model customer behavior, we need to get the data in a usable format. The lifetimes library has a great function that will do just that for us.

```python
from lifetimes.utils import summary_data_from_transaction_data
dff = summary_data_from_transaction_data(df, 'cust_id', 'order_date', 'net_sales', 
								   observation_period_end = '03-31-2019')
```


You’ll notice that the dataframe now has the following columns.


* frequency is the number of repeat purchases the customer has made.
* T represents the age of the customer, or the duration between first purchase and the end of the period.
* recency is the age of the customer when they made their most recent purchase.
* monetary_value is the average value of a customer’s purchase.

We’re now going to model future customer behavior given these variables.
> Are you interested in the theory of what we are doing? I’ve posted some resources at the end of this guide that will introduce you to lifetime value literature.


### Training and Testing the Model


First, we’ll split the data between training and test data. We’ll train the model on the training data, and then see how it performs against the test data. 

```python
from lifetimes.utils import calibration_and_holdout_data
from lifetimes import BetaGeoFitter

#Split the data between the test and training periods
summary_cal_holdout = calibration_and_holdout_data(df, 'cust_id', 'order_date',
										calibration_period_end='2018-12-31',
										observation_period_end='2019-03-31' )

#Train the model on the training (calibration) period and then see how well it
#performs on the data it has seen.
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
bgf = BetaGeoFitter()
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], 
summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)
```


![image](https://res.craft.do/user/full/46fd8ed8-eccb-4663-2d5d-ac7f63cf9d1c/doc/3B4FBB27-D0C9-4436-A74A-229424DE6782/49FDA070-982F-404D-8942-E9942EA7D9E1_2)

That looks like our model is quite effective at predicting future purchases from our customers!

Not quite sure what you see? The orange line is the what the model predicts. The blue line is how the customers actually performed. Following along on the x-axis here: 


* For users that 0 purchases in the calibration period (which was from beginning of the dataset to 12/31/18), we forecasted an average of 0.1 purchases in the holdout (test) period. The actual number was ~0.5.
* For users with 1 purchase in the calibration period, we’d expect 0.15 purchases in the holdout period. The actual number was 0.11.

And so on.

In the ideal world, you’d find an accuracy measure of our model and compare it against the accuracy of whatever crude model your org is currently running with. Most likely, someone in the org is running around with “average number of purchases in a customer lifetime” and multiplying that by number of customers to make some sort of forecast/LTV prediction.

Don’t allow that. 

### Calculating LTV


Now that we have a forecast of customer purchasing behavior, we are going to calculate lifetime value.

First, we need to verify that conditions for our Gamma-Gamma model are met. Specifically, we need to verify that there the magnitude and frequency of orders are independent.

```python
dff[['monetary_value', 'frequency']].corr()
```


Yikes! I found a moderate correlation (0.43) between them in my dataset.

**I’m going to proceed with the tutorial, but if this were real-life, I would look for another way to model lifetime value**. 

Hopefully your dataset did not have such a correlation. 

The following will calculate the value of the customer over the next 12 months, using an annual discount rate of 12.7%.

```python
from lifetimes import GammaGammaFitter

#Get rid of non-positive values
dff = dff.loc[dff['monetary_value'] > 0]

#Fit the model
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(dff['frequency'],
		dff['monetary_value'])

#Forecast lifetime value
dff['clv'] = ggf.customer_lifetime_value(
	bgf,
	dff['frequency'],
	dff['recency'],
	dff['T'],
	dff['monetary_value'],
	time=12, # months
	discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10)
```


Now we have lifetime value predictions **for each customer**. Now let’s find which marketing channel gives the highest value customers.

### Customer Value by Channel


We want to merge the value of the customer with the channel we acquired the customer by. Then, we’ll examine lifetime value by channel.

The following will give us a single dataframe that has the CLV and acquisition source of the customer: 

```python
source_table = df.drop_duplicates(subset='cust_id')
source_table = source_table[['cust_id', 'customerSource']]
merged = pd.merge(dff, source_table, on='cust_id')
merged.head()
merged['customerSource'].value_counts(normalize=True)
```


Now we can plot the distribution of lifetime values by acquisition source. I find that the distribution is important to look at because summary statistics such as median/mean can hide what may be going. 

```python
import plotly.express as px
fig = px.histogram(merged, x="clv", color="customerSource", marginal="box",
						 hover_data=merged.columns)
fig.show()
```


![image](https://res.craft.do/user/full/46fd8ed8-eccb-4663-2d5d-ac7f63cf9d1c/doc/3B4FBB27-D0C9-4436-A74A-229424DE6782/5E7785C8-AA78-4181-B240-8C53F6F8030C_2)

Visually, the different channels seem to have similar distributions. **We find that Snapchat has the highest median CLV at $112.** 

### Summary


We calculated the value of each customer and examined which of our acquisition sources gave us the best customers. Next steps would likely be to compare the lifetime value / acquistion cost (LTV/CAC) ratios between the channels and evaluate each channel’s current budget. 

CAC is the easy part, sort of. That’s a whole other post. At a minimum, the ad platforms will give you a CAC value.

You’ve done the hard part! You’ve calculated LTV for each customer.

### Resources

🤙[More of This Stuff](https://jwithing.com/?utm_source=referral&utm_medium=web&utm_campaign=github&utm_term=lifetimevaluebysource)
✍️ [Derivation of the LTV Model](http://www.brucehardie.com/notes/039/)
👶[Non-Statistician Guide to the Model](http://www.brucehardie.com/notes/032/)
🐍[Lifetimes Python Library](https://lifetimes.readthedocs.io/en/latest/)
👨‍🏫[Why You Need to Calculate CLV](https://www.youtube.com/watch?v=guj2gVEEx4s)
📊[How to Model LTV in Excel](http://www.brucehardie.com/notes/004/)

### Addendum for More Stats


OK, so it’s not really scientific to say things “look like they have the same distributions.” Nor do we want to take for granted that median CLVs are statistically significant. If you have data science resources, you may want to ask them to look into those questions for you. 

But if you had data science resources to leverage, you may not be on this page.

We’re going to perform a Kruskal-Wallis test to see if the different channels have statistically significant differences in their lifetime value distributions. If so, we can say with greater confidence that one channel gives us better customers than the others. 

I’m going to compare each channel to Facebook. Based on spending patterns, it seems like Facebook is the favored channel, so I think we’d want to express how Snapchat/TikTok perform relative to Facebook.

```python
from scipy import stats
stats.kruskal(merged['clv'].loc[merged['customerSource'] == 'facebook'],
			  merged['clv'].loc[merged['customerSource'] == 'snapchat'])
```


This test comes back with a very low p-value, which means **we can confirm that the lifetime values of users coming from Snapchat are different than Facebook users.**

```python
stats.kruskal(merged['clv'].loc[merged['customerSource'] == 'facebook'],
			  merged['clv'].loc[merged['customerSource'] == 'tiktok'])
```


This test comes back with a very low p-value, which means **we can cannot confirm that the lifetime values of users coming from TikTok are different than Facebook users.**

In summary, we’ve confirmed that we can expect higher lifetime value customers to come through Snapchat when compared to Facebook.
