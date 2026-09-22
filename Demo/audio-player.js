(() => {
  const playIcon = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.7a.7.7 0 0 1 1.05-.6l6 4.3a.74.74 0 0 1 0 1.2l-6 4.3A.7.7 0 0 1 3 10.3z"/></svg>';
  const pauseIcon = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5h2.7v9H2.5zm4.3 0h2.7v9H6.8z"/></svg>';

  const formatTime = (seconds) => {
    if (!Number.isFinite(seconds) || seconds < 0) return "--:--";
    const wholeSeconds = Math.floor(seconds);
    const minutes = Math.floor(wholeSeconds / 60);
    const remainder = String(wholeSeconds % 60).padStart(2, "0");
    return `${minutes}:${remainder}`;
  };

  function enhance(audio) {
    const label = audio.getAttribute("aria-label") || "Audio sample";
    const player = document.createElement("div");
    player.className = "audio-control";

    const top = document.createElement("div");
    top.className = "audio-control-top";

    const button = document.createElement("button");
    button.className = "audio-play-button";
    button.type = "button";
    button.innerHTML = playIcon;
    button.setAttribute("aria-label", `Play ${label}`);

    const time = document.createElement("span");
    time.className = "audio-time";
    time.textContent = "0:00 / --:--";
    time.setAttribute("aria-hidden", "true");

    const seek = document.createElement("input");
    seek.className = "audio-seek";
    seek.type = "range";
    seek.min = "0";
    seek.max = "1000";
    seek.step = "1";
    seek.value = "0";
    seek.setAttribute("aria-label", `Seek ${label}`);
    seek.setAttribute("aria-valuetext", "0:00 of --:--");

    top.append(button, time);
    player.append(top, seek);
    audio.insertAdjacentElement("beforebegin", player);
    audio.controls = false;
    audio.preload = "metadata";

    const syncProgress = () => {
      const duration = audio.duration;
      const current = audio.currentTime || 0;
      const ratio = Number.isFinite(duration) && duration > 0 ? current / duration : 0;
      seek.value = String(Math.round(ratio * 1000));
      seek.style.setProperty("--progress", `${ratio * 100}%`);
      const currentText = formatTime(current);
      const durationText = formatTime(duration);
      time.textContent = `${currentText} / ${durationText}`;
      seek.setAttribute("aria-valuetext", `${currentText} of ${durationText}`);
    };

    const syncPlayback = () => {
      const isPlaying = !audio.paused && !audio.ended;
      button.innerHTML = isPlaying ? pauseIcon : playIcon;
      button.setAttribute("aria-label", `${isPlaying ? "Pause" : "Play"} ${label}`);
    };

    button.addEventListener("click", () => {
      if (audio.paused) {
        audio.play().catch(() => {
          button.setAttribute("aria-label", `Audio unavailable: ${label}`);
        });
      } else {
        audio.pause();
      }
    });

    seek.addEventListener("input", () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        audio.currentTime = (Number(seek.value) / 1000) * audio.duration;
        syncProgress();
      }
    });

    audio.addEventListener("loadedmetadata", syncProgress);
    audio.addEventListener("durationchange", syncProgress);
    audio.addEventListener("timeupdate", syncProgress);
    audio.addEventListener("play", syncPlayback);
    audio.addEventListener("pause", syncPlayback);
    audio.addEventListener("ended", () => {
      syncPlayback();
      syncProgress();
    });
    syncProgress();
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("js-enabled");
    document.querySelectorAll("audio.audio-player").forEach(enhance);

    document.querySelectorAll("audio.audio-player").forEach((audio) => {
      audio.addEventListener("play", () => {
        document.querySelectorAll("audio.audio-player").forEach((other) => {
          if (other !== audio) other.pause();
        });
      });
    });
  });
})();
