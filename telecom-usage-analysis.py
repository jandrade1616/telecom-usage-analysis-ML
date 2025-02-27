#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cargar todas las librerías
from scipy import stats as st
import numpy as np
import pandas as pd
import math 


# In[2]:


# Carga los archivos de datos en diferentes DataFrames
users = pd.read_csv('/datasets/megaline_users.csv')
calls = pd.read_csv('/datasets/megaline_calls.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')


# In[3]:


# Imprime la información general/resumida sobre el DataFrame de las tarifas
plans = pd.read_csv('/datasets/megaline_plans.csv')
plans.info()
summary_plans = plans.describe()
print (summary_plans)


# In[4]:


# Imprime una muestra de los datos para las tarifas
print (plans.head(10))


# In[5]:


print("Nombres de las columnas:", plans.columns)
duplicates = plans.duplicated().sum()
print(f"Número de filas duplicadas: {duplicates}")
null_values = plans.isnull().sum()
print("Valores nulos en cada columna:")
print(null_values)
print("Tipos de datos:")
print(plans.dtypes)


# In[6]:


# Imprime la información general/resumida sobre el DataFrame de usuarios
users = pd.read_csv('/datasets/megaline_users.csv')
users.info()
summary_users = users.describe()
print (summary_users)


# In[7]:


# Imprime una muestra de datos para usuarios
print (users.head(10))


# In[8]:


users['reg_date'] = pd.to_datetime(users['reg_date'])
users['churn_date'] = pd.to_datetime(users['churn_date'], errors='coerce')


# In[9]:


users['is_active'] = users['churn_date'].isna()
print(users.info())
print(users.head(10))


# In[10]:


# Imprime la información general/resumida sobre el DataFrame de las llamadas
calls = pd.read_csv('/datasets/megaline_calls.csv')
calls.info()
summary_calls = calls.describe()
print (summary_calls)



# In[11]:


# Imprime una muestra de datos para las llamadas
print(calls.head(10))


# In[12]:


calls['call_date'] = pd.to_datetime(calls['call_date'])


# In[13]:


calls_with_zero_duration = calls[calls['duration'] == 0]
print(f"Número de llamadas con duración 0: {len(calls_with_zero_duration)}")
calls['is_effective'] = calls['duration'] > 0
print(calls.info())
print(calls.head(20))


# In[14]:


# Imprime la información general/resumida sobre el DataFrame de los mensajes
messages = pd.read_csv('/datasets/megaline_messages.csv')
messages.info()
summary_messages = messages.describe()
print (summary_messages)


# In[15]:


# Imprime una muestra de datos para los mensajes
print(messages.head(10))


# In[16]:


messages["message_date"] = pd.to_datetime(messages["message_date"])


# In[17]:


print(messages.info())
print(messages.head())


# In[18]:


# Imprime la información general/resumida sobre el DataFrame de internet
internet = pd.read_csv('/datasets/megaline_internet.csv')
internet.info()
summary_internet = internet.describe()
print (summary_internet)


# In[19]:


# Imprime una muestra de datos para el tráfico de internet
print(internet.head(10))


# In[20]:


internet['session_date'] = pd.to_datetime(internet['session_date'])


# In[21]:


internet_with_zero_usage = internet[internet['mb_used'] == 0]
print(f"Número de sesiones con mb_used igual a 0: {len(internet_with_zero_usage)}")
internet['spend_megas'] = internet['mb_used'] > 0
print(internet.info())
print(internet.head(10))


# In[22]:


# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
print(plans)


# In[23]:


# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
calls['month_year'] = calls['call_date'].dt.to_period('M')
calls_per_month = calls.groupby(['user_id', 'month_year']).size().reset_index(name='call_count')
print("Cantidad total de registros en calls X month:", len(calls_per_month))
calls_per_month.to_csv('calls_per_month.csv', index=False)


# In[24]:


# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
calls['month_year'] = calls['call_date'].dt.to_period('M')
minutes_per_month = calls.groupby(['user_id', 'month_year'])['duration'].sum().reset_index()
minutes_per_month = minutes_per_month.rename(columns={'duration': 'total_minutes'})
print("Cantidad total de registros en minutes_per_month:", len(minutes_per_month))
minutes_per_month.to_csv('minutes_per_month.csv', index=False)


# In[25]:


# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
messages['month_year'] = messages['message_date'].dt.to_period('M')
messages_per_month = messages.groupby(['user_id', 'month_year']).size().reset_index(name='message_count')
messages_per_month = messages.groupby(['user_id', 'month_year']).size().reset_index(name='message_count')
print("Cantidad total de registros en messages_per_month:", len(messages_per_month))
messages_per_month.to_csv('messages_per_month.csv', index=False)


# In[26]:


# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
internet['month_year'] = internet['session_date'].dt.to_period('M')
data_usage_per_month = internet.groupby(['user_id', 'month_year'])['mb_used'].sum().reset_index()
data_usage_per_month = data_usage_per_month.rename(columns={'mb_used': 'total_mb_used'})
print("Cantidad total de registros en data_usage_per_month:", len(data_usage_per_month))
data_usage_per_month.to_csv('data_usage_per_month.csv', index=False)


# In[27]:


# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month

calls_per_month['month_year'] = calls_per_month['month_year'].astype(str)
minutes_per_month['month_year'] = minutes_per_month['month_year'].astype(str)
messages_per_month['month_year'] = messages_per_month['month_year'].astype(str)
data_usage_per_month['month_year'] = data_usage_per_month['month_year'].astype(str)

merged_data = pd.merge(calls_per_month, minutes_per_month, on=['user_id', 'month_year'], how='outer')
merged_data = pd.merge(merged_data, messages_per_month, on=['user_id', 'month_year'], how='outer')
merged_data = pd.merge(merged_data, data_usage_per_month, on=['user_id', 'month_year'], how='outer')


merged_data.fillna(0, inplace=True)


print("Cantidad total de registros en merged_data:", len(merged_data))
merged_data.to_csv('merged_data.csv', index=False)



# In[28]:


# Añade la información de la tarifa
merged_data = pd.merge(merged_data, users[['user_id', 'plan']], on='user_id', how='left')
print("Columnas en merged_data antes de añadir 'plans':", merged_data.columns)
merged_data = pd.merge(merged_data, plans, left_on='plan', right_on='plan_name', how='left')
print("Columnas en merged_data después de añadir 'plans':", merged_data.columns)



# In[29]:


calls_per_month['month_year'] = calls_per_month['month_year'].astype(str)
minutes_per_month['month_year'] = minutes_per_month['month_year'].astype(str)
messages_per_month['month_year'] = messages_per_month['month_year'].astype(str)
data_usage_per_month['month_year'] = data_usage_per_month['month_year'].astype(str)


merged_data = pd.merge(calls_per_month, minutes_per_month, on=['user_id', 'month_year'], how='outer')
merged_data = pd.merge(merged_data, messages_per_month, on=['user_id', 'month_year'], how='outer')
merged_data = pd.merge(merged_data, data_usage_per_month, on=['user_id', 'month_year'], how='outer')

merged_data.fillna(0, inplace=True)

merged_data = pd.merge(merged_data, users[['user_id', 'plan']], on='user_id', how='left')
merged_data = pd.merge(merged_data, plans, left_on='plan', right_on='plan_name', how='left')
merged_data['usd_per_mb'] = merged_data['usd_per_gb'] / 1024


def calculate_monthly_revenue(row):
    extra_messages = max(row['message_count'] - row['messages_included'], 0)
    extra_mb = max(row['total_mb_used'] - row['mb_per_month_included'], 0)
    extra_minutes = max(row['total_minutes'] - row['minutes_included'], 0)
    
    extra_costs = (extra_messages * row['usd_per_message'] +
                   extra_mb * row['usd_per_mb'] +
                   extra_minutes * row['usd_per_minute'])
    
    return row['usd_monthly_pay'] + extra_costs

# Calculo de los ingresos mensuales 
merged_data['monthly_revenue'] = merged_data.apply(calculate_monthly_revenue, axis=1)
merged_data['total_minutes'] = merged_data['total_minutes'].round()
merged_data['total_mb_used'] = merged_data['total_mb_used'].round()

print("Cantidad total de registros en merged_data:", len(merged_data))
merged_data.to_csv('merged_data_with_revenue.csv', index=False)

monthly_revenue_summary = merged_data.groupby(['user_id', 'month_year'])['monthly_revenue'].sum().reset_index()
print("Cantidad total de  ingresos mensuales para cada usuario es:", len(monthly_revenue_summary))
monthly_revenue_summary.to_csv('monthly_revenue_summary.csv', index=False)




# In[30]:


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.

import matplotlib.pyplot as plt


# Crear una nueva columna 'month_year' que indique el mes y año
calls['month_year'] = calls['call_date'].dt.to_period('M')

# Añadir la información de la tarifa desde el DataFrame 'users'
calls = pd.merge(calls, users[['user_id', 'plan']], on='user_id', how='left')

# Agrupar los datos por 'plan' y 'month_year' para calcular la duración promedio de llamadas
avg_call_duration = calls.groupby(['plan', 'month_year'])['duration'].mean().reset_index()

# Convertir 'month_year' a cadena para facilitar el trazado
avg_call_duration['month_year'] = avg_call_duration['month_year'].astype(str)

# Pivotar los datos para trazarlos
pivot_table = avg_call_duration.pivot(index='month_year', columns='plan', values='duration')

# Trazar el gráfico de barras
pivot_table.plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('Duración Promedio de Llamadas por Plan y Mes')
plt.xlabel('Mes-Año')
plt.ylabel('Duración Promedio de Llamadas (min)')
plt.xticks(rotation=45)
plt.legend(title='Plan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[32]:


# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
calls['call_date'] = pd.to_datetime(calls['call_date'])
calls['month_year'] = calls['call_date'].dt.to_period('M')
calls = pd.merge(calls, users[['user_id', 'plan']], on='user_id', how='left')
monthly_minutes = calls.groupby(['user_id', 'plan', 'month_year'])['duration'].sum().reset_index()
plt.figure(figsize=(14, 8))
for plan in monthly_minutes['plan'].unique():
    subset = monthly_minutes[monthly_minutes['plan'] == plan]
    plt.hist(subset['duration'], bins=20, alpha=0.5, label=plan)

plt.title('Distribución del Número de Minutos Mensuales por Plan')
plt.xlabel('Minutos Mensuales')
plt.ylabel('Frecuencia')
plt.legend(title='Plan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[34]:


# Calcula la media y la varianza de la duración mensual de llamadas.

calls['call_date'] = pd.to_datetime(calls['call_date'])
calls['month_year'] = calls['call_date'].dt.to_period('M')
calls = pd.merge(calls, users[['user_id', 'plan']], on='user_id', how='left')
monthly_call_duration = calls.groupby(['user_id', 'plan', 'month_year'])['duration'].sum().reset_index()
mean_duration_by_plan = monthly_call_duration.groupby('plan')['duration'].mean()
variance_duration_by_plan = monthly_call_duration.groupby('plan')['duration'].var()
print("Media de la duración mensual de llamadas por plan:")
print(mean_duration_by_plan)
print("\nVarianza de la duración mensual de llamadas por plan:")
print(variance_duration_by_plan)


# In[35]:


# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
plt.figure(figsize=(14, 8))
monthly_call_duration.boxplot(column='duration', by='plan', grid=False)
plt.title('Distribución de la Duración Mensual de Llamadas por Plan')
plt.suptitle('')
plt.xlabel('Plan')
plt.ylabel('Duración Mensual de Llamadas (min)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[36]:


# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
messages['message_date'] = pd.to_datetime(messages['message_date'])
messages['month_year'] = messages['message_date'].dt.to_period('M')
messages = pd.merge(messages, users[['user_id', 'plan']], on='user_id', how='left')
monthly_message_count = messages.groupby(['user_id', 'plan', 'month_year']).size().reset_index(name='message_count')
mean_messages_by_plan = monthly_message_count.groupby('plan')['message_count'].mean()
variance_messages_by_plan = monthly_message_count.groupby('plan')['message_count'].var()


# In[37]:


print("Media del número de mensajes mensuales por plan:")
print(mean_messages_by_plan)
print("\nVarianza del número de mensajes mensuales por plan:")
print(variance_messages_by_plan)


# In[38]:


plt.figure(figsize=(14, 8))
monthly_message_count.boxplot(column='message_count', by='plan', grid=False)
plt.title('Distribución del Número de Mensajes Mensuales por Plan')
plt.suptitle('')
plt.xlabel('Plan')
plt.ylabel('Número de Mensajes Mensuales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[39]:


# Compara la cantidad de tráfico de Internet consumido por usuarios por plan
internet = pd.merge(internet, users[['user_id', 'plan']], on='user_id', how='left')
monthly_internet_usage = internet.groupby(['user_id', 'plan', 'month_year'])['mb_used'].sum().reset_index()
mean_internet_usage_by_plan = monthly_internet_usage.groupby('plan')['mb_used'].mean()
variance_internet_usage_by_plan = monthly_internet_usage.groupby('plan')['mb_used'].var()


# In[40]:


print("Media del tráfico de Internet mensual por plan:")
print(mean_internet_usage_by_plan)
print("\nVarianza del tráfico de Internet mensual por plan:")
print(variance_internet_usage_by_plan)


# In[41]:


plt.figure(figsize=(14, 8))
monthly_internet_usage.boxplot(column='mb_used', by='plan', grid=False)
plt.title('Distribución del Tráfico de Internet Mensual por Plan')
plt.suptitle('')
plt.xlabel('Plan')
plt.ylabel('Tráfico de Internet Mensual (MB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[40]:


mean_revenue_by_plan = merged_data.groupby('plan')['monthly_revenue'].mean()
variance_revenue_by_plan = merged_data.groupby('plan')['monthly_revenue'].var()


# In[41]:


print("Media de los ingresos mensuales por plan:")
print(mean_revenue_by_plan)
print("\nVarianza de los ingresos mensuales por plan:")
print(variance_revenue_by_plan)


# In[42]:


plt.figure(figsize=(14, 8))
merged_data.boxplot(column='monthly_revenue', by='plan', grid=False)
plt.title('Distribución de los Ingresos Mensuales por Plan')
plt.suptitle('')
plt.xlabel('Plan')
plt.ylabel('Ingresos Mensuales (USD)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.show()


# In[43]:


# Prueba las hipótesis
ultimate_revenue = merged_data[merged_data['plan'] == 'ultimate']['monthly_revenue']
surf_revenue = merged_data[merged_data['plan'] == 'surf']['monthly_revenue']
t_stat, p_value = st.ttest_ind(ultimate_revenue, surf_revenue, equal_var=False)
alpha = 0.05
print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_value}")
if p_value < alpha:
    print("Rechazamos la hipótesis nula: los ingresos promedio son significativamente diferentes entre los planes Ultimate y Surf.")
else:
    print("No podemos rechazar la hipótesis nula: no hay suficiente evidencia para decir que los ingresos promedio son diferentes entre los planes Ultimate y Surf.")


# In[54]:


merged_data = merged_data.merge(users[['user_id', 'city']], on='user_id', how='left')
def determine_region(city):
    if 'NY' in city or 'NJ' in city:
        return 'NY-NJ'
    else:
        return 'Other'

merged_data['region'] = merged_data['city'].apply(determine_region)

ny_nj_revenue = merged_data[merged_data['region'] == 'NY-NJ']['monthly_revenue'].dropna()
other_revenue = merged_data[merged_data['region'] == 'Other']['monthly_revenue'].dropna()

print(f"NY-NJ Revenue Data Count: {len(ny_nj_revenue)}")
print(f"Other Revenue Data Count: {len(other_revenue)}")

if len(ny_nj_revenue) == 0 or len(other_revenue) == 0:
    raise ValueError("No hay suficientes datos en una de las categorías para realizar la prueba t.")


t_stat, p_value = st.ttest_ind(ny_nj_revenue, other_revenue, equal_var=False)
alpha = 0.05


print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_value}")

if p_value < alpha:
    print("Rechazamos la hipótesis nula: los ingresos promedio son significativamente diferentes entre los usuarios del área NY-NJ y otras regiones.")
else:
    print("No podemos rechazar la hipótesis nula: no hay suficiente evidencia para decir que los ingresos promedio son diferentes entre los usuarios del área NY-NJ y otras regiones.")


