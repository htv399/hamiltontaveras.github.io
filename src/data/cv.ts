import type { Language } from "../config/site";

type CvEntry = { title: string; organization?: string; period?: string; details: string[] };
type CvSection = { id: string; title: string; entries: CvEntry[] };

export const cvData: Record<Language, { role: string; profile: string; sections: CvSection[] }> = {
  en: {
    role: "Chief Data Governance Officer & Risk Analytics",
    profile: "Economist with more than 10 years of experience in economics, finance and consulting, specializing in econometric and machine-learning models for forecasting, process optimization, risk analysis and valuation.",
    sections: [
      { id:"professional-experience", title:"Professional experience", entries:[
        {title:"Chief Data Governance Officer",organization:"Dirección General de Impuestos Internos (DGII)",period:"July 2024 – Present",details:["Leads the Institutional Data Strategy 2025–2028.","Defines standards for data management, interoperability and quality across key institutional processes."]},
        {title:"Head of Risk Analysis and Prioritization",organization:"DGII",period:"July 2019 – July 2024",details:["Implemented the Tax Risk Prioritization and Consolidation System.","Developed machine-learning models for non-compliance and anomaly detection and the Global Taxpayer Risk Profile."]},
        {title:"Economic and Tax Studies Specialist",organization:"DGII",period:"October 2014 – June 2019",details:["Conducted economic and tax research on fiscal reforms and policy measures.","Analyzed fiscal implications of tax incentives across several sectors."]}
      ]},
      { id:"consulting", title:"Consulting", entries:[
        {title:"Climate-Resilient Housing Strategy",organization:"World Bank",period:"August 2024 – June 2025",details:["Diagnosed the housing sector in Greater Santo Domingo across demand, supply, finance and institutional dimensions.","Assessed macroeconomic and climate-resilience implications for public policy."]},
        {title:"Macroeconomic Forecasting System",organization:"Ministry of Industry, Commerce and MSMEs",period:"May 2024 – December 2024",details:["Designed a hybrid machine-learning and econometric forecasting methodology.","Implemented an automated forecasting workflow using R, Python and Power BI."]},
        {title:"Tax Revenue Forecasting",organization:"Dirección General de Ingresos, Panama – CIAT",period:"March 2022 – April 2022",details:["Developed an automated econometric module of base-tax elasticities."]}
      ]},
      { id:"teaching", title:"Teaching", entries:[{title:"Economics and finance instructor",organization:"Pontificia Universidad Católica Madre y Maestra (PUCMM)",period:"January 2018 – Present",details:["Graduate courses in Financial Engineering, International Financial Markets and Corporate Valuation.","Undergraduate courses in Econometrics, Microeconometrics and International Macroeconomics."]}]},
      { id:"education", title:"Education and certifications", entries:[
        {title:"Financial Valuation Modelling Analyst (FMVA)",organization:"Corporate Finance Institute",period:"2025 – Present",details:[]},{title:"Financial Engineering Specialization",organization:"Columbia University",period:"2024 – Present",details:[]},{title:"Deep Learning Specialization",organization:"DeepLearning.AI",period:"2023",details:[]},{title:"Machine Learning Specialization",organization:"Stanford University & DeepLearning.AI",period:"2022",details:[]},{title:"M.A. in Economics",organization:"Georgetown University",period:"2017 · Magna Cum Laude",details:[]},{title:"M.A. in Economics",organization:"ILADES, Chile",period:"2017 · Magna Cum Laude",details:[]},{title:"Postgraduate in Quantitative Methods",organization:"PUCMM",period:"2015 · Cum Laude",details:[]},{title:"B.A. in Economics",organization:"UASD",period:"2013 · Magna Cum Laude",details:[]}
      ]},
      { id:"research", title:"Publications and research", entries:[
        {title:"News Shocks and Tax Reform in the Dominican Republic: a Bayesian DSGE Approach",organization:"Georgetown University & Universidad Alberto Hurtado",period:"2017",details:[]},{title:"Macroeconomic Determinants of Tax Revenues in the Dominican Republic: a SVAR Approach",organization:"DGII – Economic and Tax Studies",period:"2016",details:[]},{title:"Cost–Benefit Analysis of the Tax Incentive Regime for Free Zones in the Dominican Republic",organization:"Ministry of Finance Award · First Place",period:"2019",details:[]},{title:"Opportunities and Entrepreneurship: Evidence on Advanced Labor Market Experience",organization:"Graduate Student Research Center, University of California, Berkeley",details:[]},{title:"Determinants of Intimate Partner Violence Against Women in Dominican Households",organization:"Ministry of Economy, Planning and Development · FIES Award",period:"2020",details:[]}
      ]},
      { id:"skills", title:"Skills", entries:[{title:"Technical",details:["R, Python, SQL, PostgreSQL, EViews, Docker, Airflow, Power BI, machine learning and econometrics."]},{title:"Industry",details:["Data governance; macroeconomic, financial and tax risk; compliance; financial valuation."]},{title:"Languages",details:["Spanish and English."]}]}
    ]
  },
  es: {
    role: "Chief Data Governance Officer y analítica de riesgos",
    profile: "Economista con más de 10 años de experiencia en economía, finanzas y consultoría, especializado en modelos econométricos y de aprendizaje automático para pronóstico, optimización de procesos, análisis de riesgos y valoración.",
    sections: [
      { id:"experiencia-profesional", title:"Experiencia profesional", entries:[
        {title:"Chief Data Governance Officer",organization:"Dirección General de Impuestos Internos (DGII)",period:"Julio de 2024 – Actualidad",details:["Lidera la Estrategia Institucional de Datos 2025–2028.","Define estándares de gestión, interoperabilidad y calidad de datos en procesos institucionales clave."]},{title:"Encargado de Análisis y Priorización de Riesgos",organization:"DGII",period:"Julio de 2019 – Julio de 2024",details:["Implementó el Sistema de Priorización y Consolidación de Riesgos Tributarios.","Desarrolló modelos de aprendizaje automático para detectar incumplimiento y anomalías, así como el Perfil Global de Riesgo del Contribuyente."]},{title:"Especialista en Estudios Económicos y Tributarios",organization:"DGII",period:"Octubre de 2014 – Junio de 2019",details:["Realizó investigaciones económicas y tributarias sobre reformas fiscales y medidas de política.","Analizó implicaciones fiscales de incentivos tributarios en distintos sectores."]}
      ]},
      { id:"consultoria", title:"Consultoría", entries:[
        {title:"Estrategia de vivienda resiliente al clima",organization:"Banco Mundial",period:"Agosto de 2024 – Junio de 2025",details:["Diagnosticó el sector vivienda del Gran Santo Domingo en sus dimensiones de demanda, oferta, financiamiento e institucionalidad.","Evaluó implicaciones macroeconómicas y de resiliencia climática para la política pública."]},{title:"Sistema de pronóstico macroeconómico",organization:"Ministerio de Industria, Comercio y Mipymes",period:"Mayo de 2024 – Diciembre de 2024",details:["Diseñó una metodología híbrida de aprendizaje automático y econometría.","Implementó un flujo automatizado con R, Python y Power BI."]},{title:"Pronóstico de ingresos tributarios",organization:"Dirección General de Ingresos, Panamá – CIAT",period:"Marzo de 2022 – Abril de 2022",details:["Desarrolló un módulo econométrico automatizado de elasticidades base-impuesto."]}
      ]},
      { id:"docencia", title:"Docencia", entries:[{title:"Docente de economía y finanzas",organization:"Pontificia Universidad Católica Madre y Maestra (PUCMM)",period:"Enero de 2018 – Actualidad",details:["Cursos de posgrado en Ingeniería Financiera, Mercados Financieros Internacionales y Valoración Corporativa.","Cursos de grado en Econometría, Microeconometría y Macroeconomía Internacional."]}]},
      { id:"educacion", title:"Educación y certificaciones", entries:[
        {title:"Financial Valuation Modelling Analyst (FMVA)",organization:"Corporate Finance Institute",period:"2025 – Actualidad",details:[]},{title:"Especialización en Ingeniería Financiera",organization:"Columbia University",period:"2024 – Actualidad",details:[]},{title:"Especialización en Deep Learning",organization:"DeepLearning.AI",period:"2023",details:[]},{title:"Especialización en Machine Learning",organization:"Stanford University y DeepLearning.AI",period:"2022",details:[]},{title:"M.A. en Economía",organization:"Georgetown University",period:"2017 · Magna Cum Laude",details:[]},{title:"M.A. en Economía",organization:"ILADES, Chile",period:"2017 · Magna Cum Laude",details:[]},{title:"Posgrado en Métodos Cuantitativos",organization:"PUCMM",period:"2015 · Cum Laude",details:[]},{title:"Licenciatura en Economía",organization:"UASD",period:"2013 · Magna Cum Laude",details:[]}
      ]},
      { id:"investigacion", title:"Publicaciones e investigación", entries:[] },
      { id:"habilidades", title:"Habilidades", entries:[{title:"Técnicas",details:["R, Python, SQL, PostgreSQL, EViews, Docker, Airflow, Power BI, aprendizaje automático y econometría."]},{title:"Áreas",details:["Gobernanza de datos; riesgo macroeconómico, financiero y tributario; cumplimiento; valoración financiera."]},{title:"Idiomas",details:["Español e inglés."]}]}
    ]
  }
};

cvData.es.sections.find((section) => section.id === "investigacion")!.entries = cvData.en.sections.find((section) => section.id === "research")!.entries;
