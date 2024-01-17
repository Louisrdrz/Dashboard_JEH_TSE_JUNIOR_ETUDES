import numpy as np
import streamlit as st
import plotly.graph_objs as go  
import pandas as pd

#si un seul intervenant c'est ok, si plusieurs : le nombre de JEH est un multiple du nombre d'intervenants
def check_jeh(n_inter, n_jeh):
    ans = False 
    if n_inter == 1 :
        ans = True
    else : 
        if n_jeh % n_inter ==0 :
            ans = True
    return ans


def phase2(budget_phase, n_inter):
    n_jeh = int(budget_phase / 450) + 1  
    valeur_jeh = budget_phase / n_jeh
    
    while not check_jeh(n_inter, n_jeh):
        n_jeh += 1
        valeur_jeh = budget_phase / n_jeh
    if valeur_jeh < 80:
        # Return a special value or raise an exception
        return "insufficient_value"
    return (n_jeh, 10 * (valeur_jeh // 10))

def calculette(budget_tot, nb_phase, coef, phase_details):
    if budget_tot % 100 != 0:
        print('Le budget total n\'est pas multiple de 100')
        return
    if (coef < 0) or (coef > 1): 
        print('valeur coef impossible')
        return

    fractions = [frac for _, frac in phase_details]
    if sum(fractions) != 1:
        print('La somme des fractions n\'est pas 1')
        return

    budget_moyen_phase = [budget_tot * frac for frac in fractions]
    
    #if budget_moyen_phase[0] / phase_details[0][0] > 460:
        #return "budget_too_high"
    
    # New check for single phase and single intervenant with high budget
    if nb_phase == 1 and phase_details[0][0] == 1 and budget_moyen_phase[0] > 460:
        n_jeh = int(budget_moyen_phase[0] / 450) + 1  # Calculate the number of JEH
        valeur_jeh = budget_moyen_phase[0] / n_jeh  # Calculate the value per JEH
        return [(n_jeh, 10 * (valeur_jeh // 10))]
    
    n_jeh_total = float('inf')
    best_list = []

    for _ in range(100):
        current_list = []
        for i in range(nb_phase):
            budget = np.random.uniform(
                (budget_moyen_phase[i] / 10) * (1 - coef), 
                (budget_moyen_phase[i] / 10) * (1 + coef)
            ) * 10
            phase_result = phase2(budget, phase_details[i][0])
            if phase_result == "insufficient_value":
                # Return a special value to indicate an error
                return "insufficient_value"
            current_list.append(phase_result)

        jeh = [t[0] for t in current_list]
        if sum(jeh) < n_jeh_total:
            n_jeh_total = sum(jeh)
            best_list = current_list

    while sum([t[0] * t[1] for t in best_list]) < 0.95 * budget_tot:
        parts_value = [budget_tot * t[1] for t in phase_details]
        parts_dif = [a - b for a, b in zip(parts_value, [t[0] * t[1] for t in best_list])]
        index = parts_dif.index(max(parts_dif))
        best_list[index] = (best_list[index][0], min(450, best_list[index][1] + 20))

    return best_list


def display_optimal_distribution(result):
    phase_names = []
    phase_values = []
    jeh_counts = []
    valeur = []
    count = 0

    for i, (n_jeh, valeur_jeh) in enumerate(result):
        st.markdown(f"**Phase {i + 1}**")
        st.info(f"Nombre de JEH : {n_jeh}")
        st.success(f"Valeur par JEH : {valeur_jeh:.2f} euros")  # Formatting for decimal values
        phase_names.append(f"Phase {i + 1}")
        phase_values.append(valeur_jeh)
        jeh_counts.append(n_jeh)
        valeur.append(n_jeh*valeur_jeh)

    count =sum(valeur)

    st.markdown(f"**Coût total de la mission HT : {count} euros**")


     # Plot the graph
    
    fig = go.Figure(data=[go.Bar(x=phase_names, y=phase_values)])
    fig.update_layout(title="Prix par phase en euros", xaxis_title="Phases", yaxis_title="Prix en euros")

    # Ajouter les annotations pour le nombre de JEH
    for i, value in enumerate(phase_values):
        fig.add_annotation(x=phase_names[i], y=value,
                            text=f"JEH: {jeh_counts[i]}",
                            showarrow=False,
                            yshift=10)

    st.plotly_chart(fig, use_container_width=True)

def display_invoice_table(result):
    # Create lists to hold the data
    designations = []
    nombre_de_jeh = []
    prix_unitaire_ht = []
    total_ht = []

    # Fill in the data based on the result
    for i, (n_jeh, valeur_jeh) in enumerate(result):
        designations.append(f"Phase {i + 1}")
        nombre_de_jeh.append(n_jeh)
        prix_unitaire_ht.append(valeur_jeh)
        total_ht.append(n_jeh * valeur_jeh)

    # Add the totals
    designations.append("Nombre total de JEH")
    nombre_de_jeh.append(sum(nombre_de_jeh))
    prix_unitaire_ht.append("")
    total_ht.append(sum(total_ht))

    # Add TVA and TOTAL TTC
    designations.extend(["", "TOTAL HT", "TVA 20% (à titre indicatif)" sur les encaissements", "TOTAL TTC"])
    nombre_de_jeh.extend([""] * 5)
    prix_unitaire_ht.extend([""] * 5)
    total_ht.extend(["", total_ht[-1], total_ht[-1] * 0.2, "", total_ht[-1] * 1.2])

    # Create a DataFrame
    invoice_df = pd.DataFrame({
        'Désignation': designations,
        'Nombre de JEH': nombre_de_jeh,
        'Prix unitaire (HT)': prix_unitaire_ht,
        'TOTAL': total_ht
    })

    # Set the DataFrame's index to be the Désignation column
    invoice_df.set_index('Désignation', inplace=True)

    # Display the DataFrame as a table in Streamlit
    st.table(invoice_df)

# Initialize session state
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False

# Function to toggle the sidebar visibility
def toggle_sidebar():
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# Main function
def main():
    # Button to show/hide sidebar
    st.button("Afficher / Cacher les paramétres d'édition", on_click=toggle_sidebar)

    # Show sidebar if toggled
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.sidebar.title("Paramétre d'édition : pseudo-aléatoire")
            st.sidebar.markdown("Coefficient uniforme entre 0 et 1, 0 = constant, 1 = aléatoire, 0.2 est raisonnable")
            coef = st.sidebar.slider("Coefficient (coef)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    st.title("Dashboard - Calculatrice de JEH - TSE Junior Etudes")
    st.markdown("""Ce dashboard interactif est conçu pour faciliter l'analyse et la planification financière des missions de la TSE Junior Etudes en termes de JEH.""")

    multi = ''' Objectifs : 
    - maximiser le montant des JEH
    - minimiser le nombre de JEH
    - avoir un nombre pair par phase si le nombre d’intervenants est pair et impair si impair 
    - contrainte du budget global 
    - contrainte du budget par phase
    - les prix des JEH peuvent être différents par phase
    '''
    st.markdown(multi)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image("Logo TSE Junior Etudes.jpg", use_column_width=True)

    budget_tot = st.number_input("Entrer le budget total HT de la mission (multiple de 100)", value = 1000, max_value = 200000, min_value=100, step=100)
    nb_phase = st.number_input("Entrer le nombre de phases", min_value=1, step=1)
    coef = 0.2
    # coef : entre 0 et 1, intervale de l'uniforme, 0 = constant, 1 = [0, 2 mu] très large, 0.2 est raisonnable
    #coef = st.input("Selectionner un coefficient entre 0 et 1", 0.0, 1.0, value = 0.4)

    if nb_phase > 1:
        budget_fractions = []
        lower_bound = 0.0
        for i in range(nb_phase - 1):
            upper_bound = st.slider(f"Fraction du budget jusqu'à la phase {i + 1}", 
                                    min_value=lower_bound, max_value=1.0, value=(lower_bound + 1.0/nb_phase), step=0.01)
            budget_fractions.append(upper_bound - lower_bound)
            lower_bound = upper_bound
        budget_fractions.append(1.0 - lower_bound)  

    elif nb_phase == 1:
        budget_fractions = [1.0]  # Si il y a qu'une phase = tout le budget

    phase_details = []
    for i in range(nb_phase):
        st.subheader(f"Phase {i + 1}")
        nb_inter_phase = st.number_input(f"Nombre d'intervenants de la phase {i + 1}", min_value=1, step=1, key=f"nb_inter_phase{i}")
        # Combine le number of intervenants avec la fraction du budget correpondante 
        phase_details.append((nb_inter_phase, budget_fractions[i]))

    if st.button("Calculer"):
        result = calculette(budget_tot, nb_phase, coef, phase_details)
        if result == "insufficient_value":
            st.error("Erreur : valeur du JEH insuffisante (< 80) dans au moins une phase.")
            st.info("Conseil : augmenter la fraction du budget de cette phase.")

        elif result == "budget_too_high":
            st.error("Erreur : le budget par intervenant est trop élevé pour une seule phase.")
            st.info("Conseil : augmenter le nombre d'intervenants.")

        else:
            st.markdown("### Distribution optimale :")
            display_optimal_distribution(result)
            display_invoice_table(result)
        

if __name__ == "__main__":
    main()


