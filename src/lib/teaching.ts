// TCH-001 resource_groups.order, verbatim. render_rule: omit empty groups
// and preserve this order.
export const resourceGroupOrder = [
  { key: "slides", label: { en: "Slides", es: "Diapositivas" } },
  { key: "notes", label: { en: "Notes", es: "Notas" } },
  { key: "manual_exercises", label: { en: "Exercises", es: "Ejercicios" } },
  { key: "lab", label: { en: "Lab", es: "Laboratorio" } },
  { key: "data", label: { en: "Data", es: "Datos" } },
  { key: "code", label: { en: "Code", es: "Código" } },
  { key: "quiz", label: { en: "Quiz", es: "Prueba" } },
  { key: "bibliography", label: { en: "References", es: "Bibliografía" } }
] as const;
