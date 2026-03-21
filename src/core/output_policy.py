def apply_uncertainty_policy(label, conf, source, min_conf):
    label_txt = str(label)
    if label_txt in {"unknown", "silence"}:
        return label_txt, source, False
    if float(conf) < float(min_conf):
        src = str(source) if source else "none"
        return "uncertain", f"{src}+uncertain", True
    return label_txt, source, False
