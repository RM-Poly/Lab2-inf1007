// Table de blackjack : on glisse les cartes du paquet vers la main du joueur ou du croupier.
// Le glisser utilise les événements « pointer » (souris, doigt, stylet) : la carte suit le
// pointeur et tombe dans la zone située dessous. Un simple clic sélectionne la carte ; un clic
// sur une zone l'y dépose (repli si glisser est peu pratique).
// Chaque déplacement est envoyé à Python : { carte: "A♥", zone: "joueur" | "croupier" | "paquet" }.

const SEUIL_GLISSER = 6; // pixels avant de considérer qu'on glisse plutôt qu'on clique

function carteHtml(c, petite = false) {
  const coin = `${c.rang}<br>${c.couleur}`;
  return `<div class="carte${c.rouge ? " rouge" : ""}" data-code="${c.code}" title="${c.code}">
            <span class="coin">${coin}</span>
            <span class="centre">${c.couleur}</span>
            ${petite ? "" : `<span class="coin bas">${coin}</span>`}
          </div>`;
}

function badge(valeur, danger = false) {
  return valeur === null ? "" : `<span class="badge${danger ? " danger" : ""}">${valeur}</span>`;
}

function construire(d) {
  const croupier = d.croupier
    ? carteHtml(d.croupier) + '<div class="carte dos"></div>'
    : '<span class="aide">Glissez ici la carte visible du croupier</span>';
  const joueur = d.joueur.length
    ? d.joueur.map((c) => carteHtml(c)).join("")
    : '<span class="aide">Glissez ici les cartes de votre main</span>';
  const scoreJoueur = d.score_joueur === null
    ? "" : badge(d.score_joueur + (d.as_souple && d.score_joueur <= 21 ? " (As = 11)" : ""),
                 d.score_joueur > 21);
  const rangees = [0, 1, 2, 3].map((i) =>
    '<div class="eventail">' +
    d.paquet.slice(i * 13, i * 13 + 13)
      .map((c) => (c ? carteHtml(c, true) : '<div class="trou"></div>')).join("") +
    "</div>").join("");

  return `
    <div class="bj"><div class="tapis">
      <div class="haut">
        <div class="zone" data-zone="croupier">
          <div class="titre">Croupier ${badge(d.score_croupier)}</div>
          <div class="rangee">${croupier}</div>
        </div>
        <div class="zone" data-zone="joueur">
          <div class="titre">Votre main ${scoreJoueur}</div>
          <div class="rangee">${joueur}</div>
        </div>
      </div>
      <div class="zone paquet" data-zone="paquet">
        <div class="titre">Paquet <span class="aide">— glissez une carte vers une main,
          ou ramenez-la ici pour l'enlever</span></div>
        ${rangees}
      </div>
    </div></div>`;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data) return;

  let racine = parentElement.querySelector(".bj-racine");
  if (!racine) {
    racine = document.createElement("div");
    racine.className = "bj-racine";
    parentElement.appendChild(racine);
  }
  racine.innerHTML = construire(data);

  // Dans un shadow DOM, elementFromPoint doit être demandé à la racine fantôme.
  const surface = typeof parentElement.elementFromPoint === "function" ? parentElement : document;
  const zones = [...racine.querySelectorAll("[data-zone]")];
  const cartes = [...racine.querySelectorAll(".carte[data-code]")];
  let choisie = null; // code de la carte sélectionnée par clic

  const envoyer = (carte, zone) => setTriggerValue("deplacement", { carte, zone, t: Date.now() });
  const zoneSous = (x, y) => surface.elementFromPoint(x, y)?.closest?.("[data-zone]") ?? null;
  const surligner = (zone) => zones.forEach((z) => z.classList.toggle("survol", z === zone));
  const majCibles = () => zones.forEach((z) => z.classList.toggle("cible", choisie !== null));

  function selectionner(el) {
    cartes.forEach((x) => x.classList.remove("choisie"));
    choisie = choisie === el.dataset.code ? null : el.dataset.code;
    if (choisie) el.classList.add("choisie");
    majCibles();
  }

  cartes.forEach((el) => {
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const depart = { x: e.clientX, y: e.clientY };
      const boite = el.getBoundingClientRect();
      let fantome = null;

      const bouger = (ev) => {
        if (!fantome) {
          if (Math.hypot(ev.clientX - depart.x, ev.clientY - depart.y) < SEUIL_GLISSER) return;
          fantome = el.cloneNode(true);
          fantome.classList.add("fantome");
          fantome.style.width = `${boite.width}px`;
          fantome.style.height = `${boite.height}px`;
          racine.appendChild(fantome);
          el.classList.add("glisse");
        }
        fantome.style.left = `${ev.clientX - boite.width / 2}px`;
        fantome.style.top = `${ev.clientY - boite.height / 2}px`;
        surligner(zoneSous(ev.clientX, ev.clientY));
      };

      const lacher = (ev) => {
        window.removeEventListener("pointermove", bouger);
        window.removeEventListener("pointerup", lacher);
        window.removeEventListener("pointercancel", lacher);
        if (!fantome) {
          selectionner(el); // c'était un clic
          return;
        }
        fantome.remove();
        el.classList.remove("glisse");
        surligner(null);
        const zone = ev.type === "pointerup" ? zoneSous(ev.clientX, ev.clientY) : null;
        if (zone) envoyer(el.dataset.code, zone.dataset.zone);
      };

      window.addEventListener("pointermove", bouger);
      window.addEventListener("pointerup", lacher);
      window.addEventListener("pointercancel", lacher);
    });
  });

  zones.forEach((zone) => {
    zone.addEventListener("click", (e) => {
      if (!choisie || e.target.closest(".carte[data-code]")) return;
      envoyer(choisie, zone.dataset.zone);
    });
  });
}
