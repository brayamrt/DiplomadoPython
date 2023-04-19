##Importar librerías de uso para el modelo
import numpy as np
import pickle
import streamlit as st
import collections

##Cargar el modelo
loaded_model = pickle.load(open('/home/julian/Documentos/ProyectoClaseEspejo/finalized_model_LOG.sav','rb'))

##Logica del modelo clasificación
def clasiLogro(input_data):

    input_data_as_array = np.asarray(input_data, dtype=float)

    input_reshaped = input_data_as_array.reshape(1,-1)

    res = loaded_model.predict(input_reshaped)

    print(res[0])

    if res==0:
        return 'Su clasificación en obtención del logro es Aceptable:0'
    elif res==1:
        return 'Su clasificación en obtención del logro es Alto Grado:1'
    else: 
        return 'Su clasificación en obtención del logro es Bajo Grado:2'

    

##Activación del modelo con datos de la app
def main():

    st.markdown("<div><p style = 'text-align:center;'><img src='https://apps.corhuila.edu.co/AsistenciaCorhuila/src/LogoCorhuila.png' width='300px'></p></div>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: grey;'>Aplicación para Clasificar a Estudiantes Clase Espejo</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>Programa de Ingeniería de Sistemas</h2>", unsafe_allow_html=True)
    
    ##Igual como estan en las variables de X en el dataframe
    st.subheader('Pregunta 1')
    st.text('De acuerdo con la siguiente convención conteste la siguiente pregunta de manera numerica:\n1. 17 a 20 años\n2. 21 a 25 años\n3. mayores de 25 ')
    Edad = st.text_input('Ingrese su edad de acuerdo a la siguiente relación:')
    st.subheader('Pregunta 2')
    st.text('De acuerdo con la siguiente convención conteste la siguiente pregunta de manera numerica:\n 1.Hombre \n 2.Mujer')
    Genero = st.text_input('Indique su Genero:')
    st.subheader('Pregunta 3')
    st.text('En un rango de 1 a 10 indique su número de semestre actual')
    Semestre = st.text_input('Indique su Semestre Actual:')
    st.text('De acuerdo con la siguiente convención conteste la siguiente pregunta de manera numerica:\n0. No vive con sus padres \n1. Vive con sus padres')
    st.subheader('Pregunta 4')
    Padres = st.text_input('Actualmente vive con sus Padres?')
    st.text('De acuerdo con la siguiente convención conteste la siguiente pregunta de manera numerica: \n0. nunca\n1. casi nunca\n2. algunas veces\n3. casi siempre\n4. siempre')
    st.subheader('Pregunta 5')
    Pedag1 = st.text_input('Le gustaría que el docente emplee Sistemas de puntos o creditos por participación?')
    st.subheader('Pregunta 6')
    Pedag2 = st.text_input('Le gustaría que el docente genere trabajos colaborativos?')
    st.subheader('Pregunta 7')
    Pedag3 = st.text_input('Le gustaría que el docente adapte de materiales y actividades al entorno actual?')
    st.subheader('Pregunta 8')
    Pedag4 = st.text_input('¿Los docentes  explican con claridad las actividades que deben realizar los estudiantes?')
    st.subheader('Pregunta 9')
    Pedag5 = st.text_input('Le gustaría que los docentes  generen instancias para que entre todos/as resolvamos las dudas que surgen durante la clase espejo?')
    st.subheader('Pregunta 10')
    Pedag6 = st.text_input('Le gustaría que en la clase espejo podamos compartir nuestras experiencias y conocer a nuestros  compañeros')
    st.subheader('Pregunta 11')
    Pedag7 = st.text_input('Actualmente en la CORHUILA, Se entiende con facilidad lo que el docente expresa?')
    st.subheader('Pregunta 12')
    Pedag8 = st.text_input('Le gustaría que los docentes realicen actividades donde los/as estudiantes que poseen más dominio acerca de algún contenido pueden ayudar a quienes aún no lo han desarrollado?')
    st.subheader('Pregunta 13')
    Pedag9 = st.text_input('Le gustaría que los docentes utilicen diversos recursos para apoyar nuestro aprendizaje?')
    st.subheader('Pregunta 14')
    Pedag10 = st.text_input('Actualmente en la CORHUILA, ¿Los docentes disponen de un medio para comunicarnos con él/ella (web,mail, facebook, foro,etc.)?')
    
    clasiF = ''
    
    if st.button('Resultado de la Clasificación Logro Académico'):
        clasiF = clasiLogro([Edad, Genero, Semestre, Padres, Pedag1, Pedag2, Pedag3, Pedag4, Pedag5, Pedag6, Pedag7, Pedag8, Pedag9, Pedag10])
        
    st.success(clasiF)
    
    
if __name__ == '__main__':
    main()


