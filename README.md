Heyhey - If you want to model the vol surface of options this is the place to do it! The UI is based in streamlit which makes it great for simple use.
Just input a ticker and get a bunch of info about the options. The output is 
1. Volatility surface (you can change a heap of parameters to make it better)
   - Stochastic volatility inspired surface
   - Automatic dividend and risk free rate calculation
   - Nice UI
   - IV lookup
   - More random stuff
2. Volatility smile(you can also change a bunch of parameters to make it better here too)
   - Comparison of volatility smiles
   - More random shit
3. Greeks Analyser
   - First, second and third derivative Greeks (Incl vanna, speed, charm, etc, etc)
   - 2D plotting of greeks compared to strike
   - 3D greek surface compared to strike and expiration
Huge thanks to GD for the initial mock up!

PS: The loading times are really slow(>10sec sometimes) because I havent optimised anything really well. Dont worry it can be better.

## Notes on Data Requests

