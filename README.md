# EV-fleet-charging-schedule

We denote arrival time solution $y \in \mathbb{R^{e-s+1}}$ to represent fraction of ev fleet arriving in the charging station, where $y_i$ represents the fraction of EV fleet to arrive in time step $s+i$. We have $e,s \in \{1,2,\ldots, T\}$.
##
- gradient_descent_concrete.py: using the concrete form to update arrival time $y$.
- gradient_descent_diagram.py: using the diagram:
	- allocate arrival time for each vehicle using $y \rightarrow$ no grad
	- allocate charging 
