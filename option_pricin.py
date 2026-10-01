import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
def func_1(stck_input):
    stock=yf.Ticker(stck_input)
    history=stock.history(period='1d')
    if history.empty:
        raise ValueError("No history found or markets closed")
    S0=float(history['Close'].iloc[-1])
    expirations=stock.options
    if not expirations:
        raise ValueError("No traded options available for this stock")
    target_expiry=expirations[0]
    expiry_date=datetime.strptime(target_expiry,"%Y-%m-%d")
    today=datetime.now()
    days_to_expiry=max((expiry_date-today).days,1)
    T=days_to_expiry/365
    opt_chain=stock.option_chain(target_expiry)
    calls=opt_chain.calls.dropna(subset=['impliedVolatility'])
    atm_idx=(calls['strike']-S0).abs().idxmin()
    atm_calls=calls.loc[atm_idx]
    K=float(atm_calls['strike'])
    sigma=float(atm_calls['impliedVolatility'])
    market_price=float(atm_calls['lastPrice'])
    return S0,K,sigma,T,market_price,target_expiry
def black_scholes(S0,T,K,sigma,r=0.045,N=100000):
    d1=(np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    bs_fair_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    half_N=N//2
    z=np.random.normal(0,1,half_N)
    drift=(r-0.5*sigma**2)*T
    shock=sigma*np.sqrt(T)
    ST_pos=S0+np.exp(drift+shock*(z))
    ST_neg=S0+np.exp(drift+shock*(-z))
    payoffs=0.5*(np.maximum((ST_pos-K),0)+np.maximum((ST_neg-K),0))
    mc_fair_price = np.exp(-r * T) * np.mean(payoffs)
    return bs_fair_price, mc_fair_price
def main():
    
    ticker = input("Enter Stock Ticker ").strip().upper()
    try:

        S0,K,sigma,T,market_price,target_expiry=func_1(ticker)
        bs_price, mc_price = black_scholes(S0, T,K, sigma)
        diff=market_price-bs_price
        if abs(diff)<0.10:
            status="Fairly Priced"
        elif diff>0:
            status="Over priced"
        else:
            status="Under priced"
        print(f"Staus: {status}")
    except Exception as e:
        print(f"[-] Error: {e}")


main()
