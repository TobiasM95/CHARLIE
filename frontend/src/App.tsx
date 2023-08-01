import { PaletteColorOptions, Theme, ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import MainPageD from "./charlie-interface-desktop/MainPage";
import MainPageM from "./charlie-interface-mobile/MainPage";
import { useEffect, useRef, useState } from "react";
import Button from "@mui/material/Button";
import { Alert, Box, Stack, Typography } from "@mui/material";
import jwtDecode from "jwt-decode";
import { userAPI } from "./datastructs/UserAPI";
import gis_info from "./Settings/gis_client_id.json";
import { useScreenSize } from "./hooks/useScreenSize";

declare module "@mui/material/styles" {
  interface CustomPalette {
    customGrey: PaletteColorOptions;
  }
  interface Palette extends CustomPalette { }
  interface PaletteOptions extends CustomPalette { }
}

declare module "@mui/material/Button" {
  interface ButtonPropsColorOverrides {
    customGrey: true;
  }
}

export type THEMENAME = "DARK" | "LIGHT";
const { palette } = createTheme();
const { augmentColor } = palette;
const createColor = (mainColor: string) => augmentColor({ color: { main: mainColor } });
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    customGrey: createColor("#0058AB")
  }
});
const lightTheme = createTheme({
  palette: {
    mode: "light",
    customGrey: createColor("#0058AB")
  }
});

interface IUserObject {
  aud: string;
  azp: string;
  email: string;
  email_verified: boolean;
  exp: number;
  family_name: string;
  given_name: string;
  iat: number;
  iss: string;
  jti: string;
  name: string;
  nbf: number;
  picture: string;
  sub: string;
}

function App() {
  const [appTheme, setAppTheme] = useState<Theme>(darkTheme);

  const divRef = useRef(null);
  const [userObject, setUserObject] = useState<IUserObject | undefined>();
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [hasUserAccess, setHasUserAccess] = useState<boolean | undefined>(undefined);
  const [hasUserRequestedAccess, setHasUserRequestedAccess] = useState<boolean>(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const mediaQueryResult = useScreenSize();

  const changeAppTheme = (themeName: THEMENAME) => {
    if (themeName === "DARK") {
      setAppTheme(darkTheme);
      return;
    }
    if (themeName === "LIGHT") {
      setAppTheme(lightTheme);
      return;
    }
    setAppTheme(lightTheme);
  };

  function skipLogin() {
    setUserObject({
      email: "john.doe@gmail.com",
      email_verified: true,
      given_name: "John",
      sub: "000000000000000000000",
      aud: "",
      azp: "",
      exp: 0,
      family_name: "",
      iat: 0,
      iss: "",
      jti: "",
      name: "John Doe",
      nbf: 0,
      picture: ""
    });
    setIsLoggedIn(true);
    setHasUserAccess(true);
  }

  function handleCallbackResponse(response: google.accounts.id.CredentialResponse) {
    handleLogIn(jwtDecode(response.credential));
  }

  function handleLogIn(userObjectRaw: any) {
    if (!userObjectRaw) {
      setUserObject(undefined);
      setIsLoggedIn(true);
      setHasUserAccess(false);
    }
    const userObjectDecoded: IUserObject = userObjectRaw;
    if (userObjectDecoded && userObjectDecoded.email_verified === true) {
      setUserObject(userObjectDecoded);
      checkIfUserHasAccess(userObjectDecoded);
      setIsLoggedIn(true);
    } else {
      setUserObject(undefined);
      setIsLoggedIn(true);
      setHasUserAccess(false);
    }
  }

  function checkIfUserHasAccess(userObject: IUserObject) {
    async function checkUserAccessAPI(email: string) {
      const email_clean: string = email.replaceAll(/\W/g, "");
      try {
        const hasAccess: boolean = await userAPI.checkAccess(email_clean);
        setError("");
        setHasUserAccess(hasAccess);
      } catch (e) {
        if (e instanceof Error) {
          setError(e.message);
          console.log(e);
          setHasUserAccess(false);
        }
      }
    }
    console.log(userObject);
    checkUserAccessAPI(userObject.email);
  }

  function handleLogOut() {
    setIsLoggedIn(false);
    setHasUserAccess(false);
  }

  function requestAccess(userObject: IUserObject) {
    async function requestUserAccess(email: string) {
      const email_clean: string = email.replaceAll(/\W/g, "");
      try {
        await userAPI.requestAccess(email_clean);
        setError("");
      } catch (e) {
        if (e instanceof Error) {
          setError(e.message);
          console.log(e);
        }
      } finally {
        setHasUserRequestedAccess(true);
      }
    }
    requestUserAccess(userObject.email);
  }

  useEffect(() => {
    if (!isLoggedIn) {
      window.google.accounts.id.initialize({
        client_id: gis_info.client_id,
        callback: handleCallbackResponse
      });

      if (divRef.current) {
        window.google.accounts.id.renderButton(divRef.current, {
          theme: "filled_black",
          size: "large",
          type: "standard",
          shape: "rectangular",
          text: "signin_with"
        });
      }
    }
  }, [isLoggedIn]);

  return (
    <ThemeProvider theme={appTheme}>
      <CssBaseline />
      {isLoggedIn && hasUserAccess === true && userObject && mediaQueryResult && mediaQueryResult["isMobile"] == false && (
        <div className="container">
          <MainPageD
            changeAppTheme={changeAppTheme}
            logOutFunc={handleLogOut}
            userFirstName={userObject.given_name}
            userSUB={userObject.sub}
          />
        </div>
      )}
      {isLoggedIn && hasUserAccess === true && userObject && mediaQueryResult && mediaQueryResult["isMobile"] == true && (
        <div className="container">
          <MainPageM
            changeAppTheme={changeAppTheme}
            logOutFunc={handleLogOut}
            userFirstName={userObject.given_name}
            userSUB={userObject.sub}
          />
        </div>
      )}
      {isLoggedIn && hasUserAccess === false && userObject && (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
        >
          <Stack>
            <Typography align="center">You do not have access to charlie yet.</Typography>
            <Button
              onClick={() => {
                requestAccess(userObject);
              }}
              variant="outlined"
            >
              Request access
            </Button>
            {hasUserRequestedAccess && <Alert severity="success">Access requested!</Alert>}
          </Stack>
        </Box>
      )}
      {!isLoggedIn && (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
        >
          <Stack>
            <Typography align="center">Please log in to chat with Charlie</Typography>
            <div ref={divRef}></div>
            <Button
              variant="outlined"
              onClick={skipLogin}
            >
              Skip login
            </Button>
          </Stack>
        </Box>
      )}
    </ThemeProvider>
  );
}

export default App;
