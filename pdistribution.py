import streamlit as st
import math
import pandas as pd

st.markdown('<span style="color:blue;font-size:40px;">Probability Distributions Calculator</span>',unsafe_allow_html=True)
tablist =["**Probability Calculator**","**Binomial**", "**Poisson**","**Normal**","**Exponential**"]
tab0,tab1, tab2, tab3, tab4 = st.tabs(tablist)

with tab0:
    st.image("poisson.png")

## Display results
def display(res1,res2,res3,res4,res5,res6):
    
    c11,c12 = st.columns(2)
    c11.write(f'**P(X={x})={res1}**')
    c12.write(f'**P(X<{x})={res2}**')
       
    c21,c22 = st.columns(2)
    c21.write(f'**P(X<={x})={res3}**')
    c22.write(f'**P(X>{x})={res4}**')

    c31,c32 = st.columns(2)
    c31.write(f'**P(X>={x})={res5}**')
    res = pd.DataFrame(res6,index=[''])
    st.write(res)
    
## tab1 (Binomial distribution)
with tab1:
    with st.form("binom_form"):
        col1,col2 = st.columns(2)
        col1.markdown('<span style="color:green;font-size:25px;">Binomial Distribution Calculator</span>',unsafe_allow_html=True)
        col2.latex(r"P(X=x)=\binom{n}{x} p^x q^{n-x}")
        
        x = st.number_input('**Binomial random variable (x)**',min_value=0,value=1)
        n = st.number_input('**Number of trials (n)**',min_value=1,value=1)
        p = st.number_input('**Probability of success in single trial (p)**',min_value=0.0,max_value=1.0,value=0.5)
        x,n,p = int(x),int(n),float(p)
        calc = st.form_submit_button("Calculate")
    class Binomial:
        """
        compute the probability of an event following Binomial distribution
        n:  the number of trials (occurrences)
        x:  the number of successful trials
        p:  probability of success in a single trial
        """
        # probability density function
        def pdf(self,x,n,p):
            q=1-p
            e1 = n-x
            p = math.comb(n,x)*(p**x)*(q**e1)
            return(p)
        
        # cumulative distribution function
        def cdf(self,x,n,p,steps=False):
            tp = 0
            terms = {}


            q=1-p
            tt = x+1
            for x in range(tt):
                e1=n-x
                tp=tp+(p**x)*math.comb(n,x)*(q**e1)
                if(steps):
                    terms[x]=(p**x)*math.comb(n,x)*(q**e1)
                
            if(steps):
                terms["Prob"]=tp
                return(terms)
            else:
                return(tp)

        # survival function
        def sf(self,x,n,p):
            return(1-self.cdf(x,n,p))
        
        # common statistics
        def stats(self,n,p):
            q=1-p
            mean = n*p
            med = math.floor(mean)
            mode = math.floor((n+1)*p)
            var = n*p*q
            skew = (q-p)/math.sqrt(n*p*q)
            kurt = (1-6*p*q)/(n*p*q)
            res = {"mean":mean,"median":med,"mode":mode,"variance":var,"skewnes":skew,"kurtosis":kurt}
            return(res)
    if(calc):
        
        bp = Binomial()
        res1 = bp.pdf(x,n,p)    # P(X=x)
        res2 = bp.cdf(x-1,n,p)  # P(X<x)
        res3 = bp.cdf(x,n,p)    # P(X<=x)
        res4 = bp.sf(x,n,p)     # P(X>x)
        res5 = bp.sf(x-1,n,p)   # P(X>=x)
        res6 = bp.stats(n,p)

        display(res1,res2,res3,res4,res5,res6)

        

# tab2 (Poisson Distribution)
with tab2:
    with st.form("pois_form"):
        col1,col2 = st.columns(2)
        col1.markdown('<span style="color:green;font-size:25px;">Poisson Distribution Calculator</span>',unsafe_allow_html=True)
        col2.latex(r"P\left( x \right) = \frac{{e^{ - \mu } \mu ^x }}{{x!}}")
        x = st.number_input('**Poisson random variable (x)**',min_value=0,value=1)
        mu = st.number_input('**Average rate of success (μ)**',min_value=1.0,value=1.0)
        x,mu = int(x),float(mu)
        calc = st.form_submit_button("Calculate")
        class Poisson:
            """
            Compute the probability of an event following Poisson distribution.

            x:  value of random variable following Poisson distribution
            mu: the average number of times an event occurs
            """
            # probability density function
            def pdf(self,x,mu):
                p = (math.exp(-mu) * (mu)**x)/math.factorial(x)
                return(p)
            
            # cumulative ditribution function
            def cdf(self,x,mu,steps=False):
                tp = 0
                terms ={}
                for x in range(x+1):
                    p = (math.exp(-mu) * (mu)**x)/math.factorial(x) 
                    if(steps):
                        terms[x]=p
                    tp = tp + p
                if(steps):
                    terms["Prob"]=tp
                    return(terms)
                else:
                    return(tp)
            
            # survival function (1-cdf)
            def sf(self,x,mu):
                return(1-self.cdf(x,mu))
            
            # common statistics
            def stats(self,mu):
                mean = mu
                med = mu + 1/3-(1/(50*mu))
                mode = math.floor(mu)
                var = mu
                skew = 1/math.sqrt(mu)
                kurt = 1/mu
                res = {"mean":mean,"median":med,"mode":mode,"variance":var,"skewnes":skew,"kurtosis":kurt}
                return(res)

    if(calc):
        pp = Poisson()
        res1 = pp.pdf(x,mu)    # P(X=x)
        res2 = pp.cdf(x-1,mu)  # P(X<x)
        res3 = pp.cdf(x,mu)    # P(X<=x)
        res4 = pp.sf(x,mu)     # P(X>x)
        res5 = pp.sf(x-1,mu)   # P(X>=x)
        res6 = pp.stats(mu)

        display(res1,res2,res3,res4,res5,res6)

## tab3 (Normal Distribution)
    
with tab3:
    with st.form("norm_form"):
        col1,col2 = st.columns(2)
        col1.markdown('<span style="color:green;font-size:25px;">Normal Distribution Calculator</span>',unsafe_allow_html=True)
        col2.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}")
        
        x = st.number_input('**Random variable (x)**',min_value=0.0,value=1.0)
        mu = st.number_input('**Average (μ)**',min_value=0.0,value=0.0)
        sd = st.number_input('**Standard deviation (σ)**',min_value=1.0,value=1.0)
        x,mu,sd = float(x),float(mu),float(sd)
        calc = st.form_submit_button("Calculate")
        class Normal:
            
            def pdf(self,x,mu=0,sd=1):
                pi = math.pi
                p = (1/(sd*math.sqrt(2*pi)))*math.exp(-1/2*((x-mu)/sd)**2)
                return(p)
            
            def sf(self,x,mu=0,sd=1):
                return(1-self.cdf(x,mu,sd))
                
            def cdf(self,x,mu=0,sd=1):
                x = (x-mu)/sd
                #'Cumulative distribution function for the standard normal distribution'
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

            def stats(self,mu=0,sd=1):
                mean = mu
                med = mu
                mode = mu
                var = sd**2
                skew = 0
                kurt = 0
                res = {"mean":mean,"median":med,"mode":mode,"variance":var,"skewnes":skew,"kurtosis":kurt}
                return(res)
    if(calc):
        np = Normal()
        res1 = np.pdf(x,mu,sd)    # P(X=x)
        res2 = np.cdf(x,mu,sd)    # P(X<x)
        res3 = np.sf(x,mu,sd)     # P(X>x)
        res4 = np.stats(mu)

        c11,c12 = st.columns(2)
        c11.write(f'**P(X={x})={res1}**')
        c12.write(f'**P(X<{x})={res2}**')

        c21,c22 = st.columns(2)
        c21.write(f'**P(X>{x})={res3}**')
        res = pd.DataFrame(res4,index=[''])
        st.write(res)

## tab4 (Exponential Distribution)
    
with tab4:
    with st.form("expon_form"):
        col1,col2 = st.columns(2)
        col1.markdown('<span style="color:green;font-size:25px;">Exponential Distribution Calculator</span>',unsafe_allow_html=True)
        col2.latex(r"f(x)=\lambda{e^{-\lambda{x}}}")
        
        x = st.number_input('**Random variable (x)**',min_value=0.1,value=1.0)
        mu = st.number_input('**Average (μ)**',min_value=0.1,value=1.0)
        x,mu= float(x),float(mu)
        calc = st.form_submit_button("Calculate")

        class Exponential:
            
            def pdf(self,x,mu):
                if(x<0):
                    return(0)
                rate = 1/mu
                p = rate*math.exp(-rate*x)
                return(p)
            
            def sf(self,x,mu):
                return(1-self.cdf(x,mu))
            
            def cdf(self,x,mu):
                rate = 1/mu
                p = 1-math.exp(-rate*x)
                return(p)
            
            def stats(self,x,mu):
                rate = 1/mu
                mean = 1/rate
                med = math.log(2)/rate
                mode = 0
                var = 1/rate**2
                skew = 2
                kurt = 6
                res={"mean":mean,"median":med,"mode":mode,"variance":var,"skewness":skew,
                    "kurtosis":kurt}
                return(res)
    if(calc):
        ep = Exponential()
        res1 = ep.pdf(x,mu)    # P(X=x)
        res2 = ep.cdf(x-1,mu)  # P(X<x)
        res3 = ep.cdf(x,mu)    # P(X<=x)
        res4 = ep.sf(x,mu)     # P(X>x)
        res5 = ep.sf(x-1,mu)   # P(X>=x)
        res6 = ep.stats(x,mu)

        display(res1,res2,res3,res4,res5,res6)

        
