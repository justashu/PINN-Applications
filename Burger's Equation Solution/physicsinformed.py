import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities

class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, u_t0):
        """Construct a PhysicsInformedBar model"""
        
        self.u_t0 = u_t0
        self.t = utilities.generate_grid_1d(10)
        self.x = utilities.generate_grid_1d(2,10,-1)
        self.X = []
        self.t0 = []
        self.x1 = []
        self.x2 = []
        for i in self.t:
            for j in self.x:
                self.X.append([i,j])
                if i == 0:
                    self.t0.append([i,j])
                if j == -1:
                    self.x1.append([i,j])
                if j == 1:
                    self.x2.append([i,j])    
        self.X = torch.tensor(self.X,requires_grad = True)
        self.x1 = torch.tensor(self.x1,requires_grad = True)
        self.x2 = torch.tensor(self.x2,requires_grad = True)
        self.t0 = torch.tensor(self.t0,requires_grad = True)
        
        self.model = utilities.build_model(2,[40,40],1)
        self.differential_equation_loss_history = None
        self.boundary_condition_loss_history = None
        self.total_loss_history = None
        self.optimizer = None

    def get_displacements(self, x):
        """Get displacements."""

        u = self.model(x)   # predict

        return u

    def costFunction(self, x, u_pred):
        """Compute the cost function."""
        u_t = utilities.get_derivative(u_pred, x, 1)[:,0].view(-1,1)
        #u_tt = utilities.get_derivative(u_pred, x,2)[:,0]
        #u_x at pts (t,-1)-> u_x_1 and (t,1) -> u_x1
        u_x1 = utilities.get_derivative(self.get_displacements(self.x1), self.x1, 1)[:,1].view(-1,1)
        u_x_1 = utilities.get_derivative(self.get_displacements(self.x2), self.x2, 1)[:,1].view(-1,1)
        u_xx = utilities.get_derivative(utilities.get_derivative(u_pred, x, 1)[:,1].view(-1,1),x,1)[:,1].view(-1,1)
         
        # Differential equation loss (f)
        differential_equation_loss = torch.sum(u_t-10**(-4)*u_xx + 5*u_pred**3 - 5*u_pred)
        differential_equation_loss = torch.sum(differential_equation_loss ** 2).view(1)

        # Boundary condition loss initialization
        boundary_condition_loss = 0

        # Sum over dirichlet and neumann boundary condition losses
        
        boundary_condition_loss += torch.sum((self.get_displacements
                                              (self.t0) - self.u_t0(self.t0[:,0],self.t0[:,1]).view(-1,1)) ** 2).view(1)
        boundary_condition_loss += torch.sum((self.get_displacements(self.x1)-self.get_displacements(self.x2)) ** 2).view(1)
        boundary_condition_loss += torch.sum(( u_x1 - u_x_1) ** 2).view(1)

        return differential_equation_loss, boundary_condition_loss

    def closure(self):
        """Calculation of training error and gradient"""
        self.optimizer.zero_grad()
        u_pred = self.get_displacements(self.X)
        loss = self.costFunction(self.X, u_pred)
        loss = loss[0] + loss[1]
        loss.backward(retain_graph=True)
        return loss

    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Set optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        elif optimizer=='LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

        # Initialize history arrays
        self.differential_equation_loss_history = np.zeros(epochs)
        self.boundary_condition_loss_history = np.zeros(epochs)
        self.total_loss_history = np.zeros(epochs)

        # Training loop
        for i in range(epochs):
            # Predict displacements
            u_pred = self.get_displacements(self.X)

            # Cost function calculation
            differential_equation_loss, boundary_condition_loss = self.costFunction(self.X, u_pred)

            # Total loss
            total_loss = differential_equation_loss + boundary_condition_loss

            # Add energy values to history
            self.differential_equation_loss_history[i] += differential_equation_loss
            self.boundary_condition_loss_history[i] += boundary_condition_loss
            self.total_loss_history[i] += total_loss

            # Print training state
            self.print_training_state(i, epochs)

            # Update parameters (Neural network train)
            self.optimizer.step(self.closure)

    def print_training_state(self, epoch, epochs, print_every=100):
        """Print the loss values of the current epoch in a training loop."""

        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tDifferential equation loss = {:2f}\t\tBoundary condition loss = {:2f}\t\tTotal loss = {:2f}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.differential_equation_loss_history[epoch],
                                self.boundary_condition_loss_history[epoch], self.total_loss_history[epoch]))

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        ax.plot(self.total_loss_history, 'k', linewidth=2, label="Total cost")
        ax.plot(self.differential_equation_loss_history, color='silver', linestyle='--', linewidth=2, label="Differential equation loss")
        ax.plot(self.boundary_condition_loss_history, color='r', linestyle='-.', linewidth=2, label="Boundary condition loss")
        

        ax.legend()
        fig.tight_layout()
        plt.show() 
        
        
