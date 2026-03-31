# Design System: The Clinical Sanctuary

## 1. Overview & Creative North Star: "The Empathetic Authority"
This design system moves away from the sterile, cold interfaces typically found in medical technology. Our Creative North Star is **"The Empathetic Authority."** We aim to balance clinical precision with a humane, tactile warmth. 

To break the "template" look, we reject the rigid 12-column grid in favor of **Intentional Asymmetry**. Significant information should feel anchored, while secondary assistive elements float with breathing room. We utilize overlapping layers and an editorial typography scale to ensure the platform feels like a high-end, bespoke tool rather than a generic dashboard.

---

## 2. Colors & Tonal Depth
The palette is rooted in a "Warm Clinical" aesthetic. We use soft neutrals to reduce eye strain and teals to evoke confidence and sterility without the "coldness" of pure blue.

### The Palette
- **Primary (`#005f6a`):** Our Deep Teal. Represents stability and professional medical oversight.
- **Secondary (`#406374`):** Ice-Blue Slate. Used for secondary actions and calming data visualizations.
- **Tertiary (`#844600`):** Alert Amber. Reserved for warnings that require attention but not panic.
- **Error (`#ba1a1a`):** Emergency Red. High-contrast for critical medical alerts.
- **Neutral Base (`#faf9f5`):** A warm, off-white surface that feels more humane than a pure `#ffffff`.

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to section off the UI. Separation must be achieved through background shifts. For example, a `surface-container-low` (`#f5f4f0`) sidebar sitting against a `surface` (`#faf9f5`) main stage.

### Surface Hierarchy & Nesting
Treat the UI as physical layers of frosted glass and premium paper.
- **Background:** `surface` (`#faf9f5`)
- **Main Content Area:** `surface-container-low` (`#f5f4f0`)
- **Floating Cards:** `surface-container-lowest` (`#ffffff`)
- **In-Card Details:** `surface-container-high` (`#e9e8e4`)

### The Glass & Gradient Rule
For assistive overlays or floating navigation, use **Glassmorphism**. Apply a semi-transparent `surface` color with a `backdrop-blur` of 12px. For primary CTAs, use a subtle linear gradient from `primary` (`#005f6a`) to `primary_container` (`#007a87`) to give the button "soul" and a tactile, slightly convex feel.

---

## 3. Typography: Editorial Precision
We pair **Manrope** (Display/Headlines) for its modern, geometric confidence with **Public Sans** (Body/Labels) for its neutral, highly legible characteristics.

- **Display-LG (Manrope, 3.5rem):** Used for primary patient names or critical status numbers.
- **Headline-MD (Manrope, 1.75rem):** For section headers. Use tight letter-spacing (-0.02em) to look "Editorial."
- **Title-MD (Public Sans, 1.125rem):** For card headers. Always use `on_surface` color for high authority.
- **Body-LG (Public Sans, 1rem):** The workhorse for communication transcripts. Ensure a line height of at least 1.6 for maximum readability in high-stress medical environments.

---

## 4. Elevation & Depth
Depth in this system is a product of light and layering, never heavy shadows.

- **The Layering Principle:** Instead of shadows, stack surface tiers. A `surface-container-lowest` card placed on a `surface-container-low` background creates a natural "lift" through color value alone.
- **Ambient Shadows:** If an element must float (e.g., a modal), use a diffused shadow: `box-shadow: 0 12px 40px rgba(27, 28, 26, 0.05)`. The shadow color uses the `on_surface` token at 5% opacity, mimicking natural light.
- **The Ghost Border Fallback:** If a boundary is required for accessibility, use the `outline_variant` (`#bdc8cb`) at **15% opacity**. High-contrast, 100% opaque borders are strictly forbidden.

---

## 5. Components

### Cards & Communication Modules
- **Style:** No borders. Use `surface-container-lowest` with a `lg` (1rem) corner radius.
- **Spacing:** Use `spacing-6` (2rem) internal padding to ensure content doesn't feel "medicalized" or cramped.
- **Separation:** Never use divider lines. Use `spacing-4` (1.4rem) of vertical white space to separate entries.

### Primary Buttons
- **Shape:** `full` (9999px) pill shape for a friendly, humane touch.
- **Color:** Gradient of `primary` to `primary_container`. 
- **Typography:** `label-md` (Bold).

### Status Indicators (Assistive Chips)
- **Warning:** `tertiary_container` background with `on_tertiary_fixed_variant` text.
- **Critical:** `error_container` background with `on_error_container` text.
- **Interaction:** These should be static or "Action Chips" with a `md` (0.75rem) radius.

### Input Fields
- **Style:** "Soft Underline" or "Subtle Fill." Use `surface_container_highest` as a background fill with no border. Upon focus, transition the background to `surface_bright` and add a `primary` 2px bottom-only border.

### Assistive Overlays (Glassmorphic)
- For tooltips or quick-actions, use a semi-transparent `secondary_container` with a `20px` backdrop blur. This keeps the medical context visible underneath while bringing the tool to the foreground.

---

## 6. Do’s and Don’ts

### Do
- **Do** use intentional asymmetry. Place a primary data point off-center to create a modern, editorial feel.
- **Do** use the `spacing-12` (4rem) and `spacing-16` (5.5rem) tokens generously. Negative space is a "trust" signal in high-end design.
- **Do** ensure all text hits a minimum 4.5:1 contrast ratio against its specific surface container.

### Don't
- **Don't** use pure black (`#000000`). Use `on_surface` (`#1b1c1a`) to keep the interface soft.
- **Don't** use "Drop Shadows" on buttons. Let the color and shape do the work.
- **Don't** use "Alert Red" for anything other than a life-critical emergency. For general errors, use the amber `tertiary` scale to maintain a "calm" environment.
- **Don't** use purple. It breaks the clinical, ice-blue/teal trust established in the primary palette.