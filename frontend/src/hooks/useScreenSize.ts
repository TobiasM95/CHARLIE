import useMediaQuery from "@mui/material/useMediaQuery";
import { mobileBreakpoint } from "../Settings/Constants";

interface ScreenSize {
    isMobile: boolean;
}

export const useScreenSize = (): ScreenSize => {
    return({
        isMobile: useMediaQuery(`(max-width: ${mobileBreakpoint})`)
    });
}