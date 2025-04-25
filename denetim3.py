import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from control import TransferFunction, step_response
import math

def format_polynomial(coeffs):
    if len(coeffs) == 0:
        return "0"
    
    terms = []
    for i, coef in enumerate(coeffs):
        power = len(coeffs) - i - 1
        
        if abs(coef) < 1e-10:
            continue
            
        if power == 0:
            term = f"{coef:.2f}"
        elif power == 1:
            term = f"{coef:.2f}s"
        else:
            term = f"{coef:.2f}s^{power}"
            
        terms.append(term)
    
    if not terms:
        return "0"
    return " + ".join(terms)

def calculate_zeta_and_wn(poles):
    """
    Calculate damping ratio (zeta) and natural frequency (wn) from poles
    """
    # Check for complex conjugate poles (second-order system)
    complex_poles = [p for p in poles if abs(p.imag) > 1e-10]
    if len(complex_poles) >= 2:
        # Use first complex pole
        pole = complex_poles[0]
        sigma = abs(pole.real)
        omega = abs(pole.imag)
        
        wn = math.sqrt(sigma**2 + omega**2)
        if wn > 0:
            zeta = sigma / wn
        else:
            zeta = 1.0
        
        return zeta, wn
    
    # For real poles
    if len(poles) >= 2:
        # Sort poles by real part magnitude
        sorted_poles = sorted([(abs(p.real), p) for p in poles], key=lambda x: x[0])
        p1_val, p1 = sorted_poles[0]
        p2_val, p2 = sorted_poles[1]
        
        # If both poles are real
        if abs(p1.imag) < 1e-10 and abs(p2.imag) < 1e-10:
            a = -p1.real
            b = -p2.real
            if a > 0 and b > 0:  # Stable system check
                wn = math.sqrt(a * b)
                zeta = (a + b) / (2 * wn) if wn > 0 else 1.0
                return zeta, wn
    
    # Single pole or other cases
    if len(poles) > 0:
        if abs(poles[0].imag) < 1e-10:  # Real pole
            wn = abs(poles[0].real)
            zeta = 1.0  # Assume critically damped
        else:  # Complex pole
            sigma = abs(poles[0].real)
            omega = abs(poles[0].imag)
            wn = math.sqrt(sigma**2 + omega**2)
            zeta = sigma / wn if wn > 0 else 1.0
        
        return zeta, wn
    
    return 0.0, 0.0  # Default values

def determine_system_type(den):
    """
    Determine system type (number of poles at s=0)
    """
    # Find number of s=0 poles
    s_zero_poles = 0
    
    if len(den) > 0:
        for i in range(len(den)-1, -1, -1):
            if abs(den[i]) < 1e-10:
                s_zero_poles += 1
            else:
                break
    
    return s_zero_poles

def determine_response_type(zeta):
    """
    Determine response type based on damping ratio
    """
    if zeta > 1.0 + 1e-10:
        return "Overdamped"
    elif abs(zeta - 1.0) <= 1e-10:
        return "Critically Damped"
    elif zeta >= 0:
        return "Underdamped"
    else:
        return "Unstable"

def main():
    st.set_page_config(page_title="Transfer Function Stability Analysis", layout="wide")
    
    st.title("Transfer Function Stability Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("System Parameters")
        
        # Create tabs for coefficient input
        tab1, tab2 = st.tabs(["Coefficients", "System Info"])
        
        with tab1:            
            # Numerator coefficients
            st.subheader("Numerator Coefficients")
            num_coeff = []
            for i in range(4):  # Up to 3rd order
                power = 3 - i
                coeff = st.number_input(f"b{power} (s^{power})", 
                                       value=1.0 if power == 0 else 0.0,
                                       step=0.1, format="%.2f", key=f"num_{power}")
                num_coeff.append(coeff)
            
            # Remove leading zeros
            while len(num_coeff) > 1 and abs(num_coeff[0]) < 1e-10:
                num_coeff.pop(0)
            
            # Denominator coefficients
            st.subheader("Denominator Coefficients")
            den_coeff = []
            for i in range(4):  # Up to 3rd order
                power = 3 - i
                default_value = 0.0
                if power == 2:  # s^2 coefficient
                    default_value = 1.0
                elif power == 1:  # s^1 coefficient
                    default_value = 2.0
                elif power == 0:  # s^0 coefficient
                    default_value = 1.0
                
                coeff = st.number_input(f"a{power} (s^{power})", 
                                       value=default_value,
                                       step=0.1, format="%.2f", key=f"den_{power}")
                den_coeff.append(coeff)
            
            # Remove leading zeros
            while len(den_coeff) > 1 and abs(den_coeff[0]) < 1e-10:
                den_coeff.pop(0)
            
            # Ensure denominator is not zero
            if len(den_coeff) == 0 or all(abs(d) < 1e-10 for d in den_coeff):
                den_coeff = [1.0]
            
            # Display transfer function
            tf_string = f"G(s) = {format_polynomial(num_coeff)} / ({format_polynomial(den_coeff)})"
            st.markdown(f"### Transfer Function\n${tf_string}$")
        
        with tab2:
            if 'num_coeff' in locals() and 'den_coeff' in locals():
                # Create transfer function
                sys = TransferFunction(num_coeff, den_coeff)
                
                # Get poles and zeros
                tf_scipy = signal.TransferFunction(num_coeff, den_coeff)
                poles = tf_scipy.poles
                zeros = tf_scipy.zeros
                
                # Check stability
                is_stable = all(pole.real < 0 for pole in poles)
                stability = "Stable" if is_stable else "Unstable"
                stability_color = "green" if is_stable else "red"
                
                st.markdown(f"### Stability\n <span style='color:{stability_color};font-weight:bold'>{stability}</span>", unsafe_allow_html=True)
                
                # Calculate zeta and wn
                zeta, wn = calculate_zeta_and_wn(poles)
                
                # Determine system type and response type
                system_type = determine_system_type(den_coeff)
                response_type = determine_response_type(zeta)
                
                # Display system parameters
                st.subheader("System Parameters")
                st.write(f"Damping Ratio (ζ): {zeta:.4f}")
                st.write(f"Natural Frequency (ωn): {wn:.4f} rad/s")
                st.write(f"System Type: Type {system_type}")
                st.write(f"Response Type: {response_type}")
                
                # Display poles and zeros
                st.subheader("Poles and Zeros")
                poles_text = "Poles:\n"
                for i, pole in enumerate(poles):
                    poles_text += f"  p{i+1} = {pole:.4f}\n"
                
                zeros_text = "Zeros:\n"
                if len(zeros) > 0:
                    for i, zero in enumerate(zeros):
                        zeros_text += f"  z{i+1} = {zero:.4f}\n"
                else:
                    zeros_text += "  None\n"
                
                st.text(poles_text + "\n" + zeros_text)
    
    with col2:
        if 'num_coeff' in locals() and 'den_coeff' in locals():
            # Create tabs for different plots
            plot_tab1, plot_tab2, plot_tab3 = st.tabs(["Step Response", "Pole-Zero Map", "Bode Plot"])
            
            with plot_tab1:
                # Step response
                t, y = step_response(sys, np.linspace(0, 20, 1000))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(t, y)
                ax.set_title('Step Response')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True)
                
                # Add stability information
                info_text = f"Status: {stability}\nζ = {zeta:.4f}, ωn = {wn:.4f} rad/s"
                ax.text(0.05, 0.95, info_text, 
                        transform=ax.transAxes, 
                        fontsize=10, fontweight='bold', color=stability_color, 
                        verticalalignment='top')
                
                st.pyplot(fig)
            
            with plot_tab2:
                # Pole-zero map
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot poles
                ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10, label='Poles')
                
                # Plot zeros
                if len(zeros) > 0:
                    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10, label='Zeros')
                
                # Imaginary axis (stability boundary)
                ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                ax.grid(True)
                ax.set_title('Pole-Zero Map')
                ax.set_xlabel('Real Axis')
                ax.set_ylabel('Imaginary Axis')
                
                # Unit circle
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta), 'g--', alpha=0.5)
                ax.axis('equal')
                ax.legend()
                
                st.pyplot(fig)
            
            with plot_tab3:
                # Bode plot
                w, mag, phase = signal.bode(signal.TransferFunction(num_coeff, den_coeff))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Magnitude plot
                ax1.semilogx(w, mag)
                ax1.set_title('Bode Diagram - Magnitude')
                ax1.set_xlabel('Frequency [rad/s]')
                ax1.set_ylabel('Magnitude [dB]')
                ax1.grid(True)
                
                # Phase plot
                ax2.semilogx(w, phase)
                ax2.set_title('Bode Diagram - Phase')
                ax2.set_xlabel('Frequency [rad/s]')
                ax2.set_ylabel('Phase [degrees]')
                ax2.grid(True)
                
                fig.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()